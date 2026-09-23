/**
 * The fake Java VM of fake_jni.h. Refer to that file for the checks.
 *
 * A local reference is a Ref record. The record stays in memory until
 * reset(), thus a use after its call ends gives a message. The heap and the
 * record list have one mutex, because a second thread (the stop request of
 * the thread fuzzer) has its own JNIEnv.
 */
#include "fake_jni.h"

#include "crash_input.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <type_traits>
#include <utility>

/** The stack printer of the sanitizer runtimes. It is null in a build without a sanitizer. */
extern "C" __attribute__((weak)) void __sanitizer_print_stack_trace();

namespace fakejni {

namespace {

/** One local reference. */
struct Ref {
    Obj *    obj    = nullptr;
    uint64_t serial = 0;     // The serial number of the call that made it.
    bool     live   = true;
    bool     native = false; // True when the native code made it, false for an argument.
};

/** One pinned buffer: string characters or array elements. */
struct Pin {
    void * ptr    = nullptr;
    Obj *  obj    = nullptr;
    bool   is_utf = false;
};

/** One native call on a thread. */
struct Frame {
    const char *       name   = "";
    uint64_t           serial = 0;
    std::vector<Ref *> refs;
    std::vector<Pin>   pins;
    size_t             native_live = 0;
    size_t             peak        = 0;
    size_t             ensured     = 16;
};

/** The JNIEnv of one thread and its state. The function table reads the state through a cast. */
struct ThreadEnv : public JNIEnv {
    std::thread::id    owner;
    std::vector<Frame> frames;
    Obj *              pending   = nullptr;
    int                countdown = -1;
    uint64_t           injected  = 0;
};

using Table = std::remove_const_t<std::remove_pointer_t<decltype(std::declval<JNIEnv>().functions)>>;

/** A method of an object: toString of a Throwable. */
struct MethodEntry {
    std::string  class_name;
    std::string  name;
    std::string  sig;
    StaticMethod fn;
    bool         is_static = false;
};

std::mutex                                       g_mutex;
std::vector<std::unique_ptr<Obj>>                g_heap;
std::vector<std::unique_ptr<Ref>>                g_refs;
std::map<std::string, Obj *>                     g_classes;
std::map<std::string, std::unique_ptr<MethodEntry>> g_methods;
uint64_t                                         g_serial = 0;

/** The heap owns the object. */
Obj * own(std::unique_ptr<Obj> obj) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_heap.push_back(std::move(obj));
    return g_heap.back().get();
}

ThreadEnv & state(JNIEnv * env, const char * fn) {
    if (env == nullptr) {
        fail("JNI %s: the JNIEnv is null", fn);
    }
    auto * t = static_cast<ThreadEnv *>(env);
    if (t->owner != std::this_thread::get_id()) {
        fail("JNI %s: a JNIEnv of a different thread was used", fn);
    }
    return *t;
}

Frame & frame(ThreadEnv & t, const char * fn) {
    if (t.frames.empty()) {
        fail("JNI %s: a JNI call outside of a native call", fn);
    }
    return t.frames.back();
}

/**
 * The start of each JNI function: the thread, the open call, and the rule
 * that only a small set of functions runs with a pending exception.
 */
ThreadEnv & enter(JNIEnv * env, const char * fn, bool allowed_with_exception = false) {
    ThreadEnv & t = state(env, fn);
    frame(t, fn);
    if (t.pending != nullptr && !allowed_with_exception) {
        if (!relaxed("pending")) {
            fail("JNI %s: called with a pending exception %s: %s (CheckJNI aborts the app here)", fn,
                 t.pending->class_name.c_str(), t.pending->message.c_str());
        }
        // The check is relaxed: the first report of each function goes to stderr, and the call runs as
        // ART runs it without CheckJNI.
        static std::mutex seen_mutex;
        static std::set<std::string> seen;
        std::lock_guard<std::mutex> lock(seen_mutex);
        if (seen.insert(fn).second) {
            fprintf(stderr, "==FAKEJNI== relaxed: JNI %s called with a pending exception %s\n", fn,
                    t.pending->class_name.c_str());
            if (__sanitizer_print_stack_trace != nullptr) {
                __sanitizer_print_stack_trace();
            }
        }
    }
    return t;
}

/** True when the fault plan makes this allocating call fail. The OutOfMemoryError is then pending. */
bool inject(ThreadEnv & t, const char * fn) {
    if (t.countdown < 0) {
        return false;
    }
    if (t.countdown > 0) {
        t.countdown -= 1;
        return false;
    }
    t.countdown = -1;
    t.injected += 1;
    throw_exception(&t, "java/lang/OutOfMemoryError", std::string("injected fault in ") + fn);
    return true;
}

Obj * deref(ThreadEnv & t, jobject ref, const char * fn, bool nullable) {
    (void) t;
    if (ref == nullptr) {
        if (!nullable) {
            fail("JNI %s: null object where JNI requires one", fn);
        }
        return nullptr;
    }
    const auto * r = reinterpret_cast<const Ref *>(ref);
    if (!r->live) {
        fail("JNI %s: the local reference %p is not live (deleted, or its call ended)", fn, (const void *) ref);
    }
    return r->obj;
}

Obj * deref_kind(ThreadEnv & t, jobject ref, const char * fn, Kind kind) {
    Obj * o = deref(t, ref, fn, false);
    if (o->kind != kind) {
        fail("JNI %s: object of the wrong type (%s)", fn, o->class_name.c_str());
    }
    return o;
}

jobject make_local(ThreadEnv & t, Obj * obj, bool native) {
    if (obj == nullptr) {
        return nullptr;
    }
    Frame & f = frame(t, "new local reference");
    auto ref = std::make_unique<Ref>();
    ref->obj    = obj;
    ref->serial = f.serial;
    ref->native = native;
    Ref * raw = ref.get();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_refs.push_back(std::move(ref));
    }
    f.refs.push_back(raw);
    if (native) {
        f.native_live += 1;
        f.peak = std::max(f.peak, f.native_live);
        if (f.native_live > f.ensured && !relaxed("capacity")) {
            fail("JNI %s: %zu live local references, more than the %zu that the call ensured (JNI guarantees 16)",
                 f.name, f.native_live, f.ensured);
        }
    }
    return reinterpret_cast<jobject>(raw);
}

/** Decode modified UTF-8 as ART does for NewStringUTF, with the checks of CheckJNI. */
bool from_modified_utf8(const char * s, std::u16string & out, std::string & why) {
    const auto * p = reinterpret_cast<const unsigned char *>(s);
    while (*p != 0) {
        const unsigned char c = *p;
        if (c < 0x80) {
            out.push_back(c);
            p += 1;
            continue;
        }
        int extra = 0;
        uint32_t cp = 0;
        if ((c & 0xE0) == 0xC0) {
            extra = 1;
            cp = c & 0x1F;
        } else if ((c & 0xF0) == 0xE0) {
            extra = 2;
            cp = c & 0x0F;
        } else if ((c & 0xF8) == 0xF0) {
            extra = 3;
            cp = c & 0x07;
        } else {
            why = "illegal start byte 0x" + std::to_string(c);
            return false;
        }
        for (int i = 1; i <= extra; ++i) {
            if ((p[i] & 0xC0) != 0x80) {
                why = "illegal continuation byte";
                return false;
            }
            cp = (cp << 6) | (p[i] & 0x3F);
        }
        p += 1 + extra;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            out.push_back((char16_t) (0xD800 + (cp >> 10)));
            out.push_back((char16_t) (0xDC00 + (cp & 0x3FF)));
        } else {
            out.push_back((char16_t) cp);
        }
    }
    return true;
}

// --- The JNI functions. Each one does the checks of CheckJNI that apply to it. ---

jint fn_EnsureLocalCapacity(JNIEnv * env, jint capacity) {
    ThreadEnv & t = enter(env, "EnsureLocalCapacity");
    if (capacity < 0) {
        fail("JNI EnsureLocalCapacity: negative capacity %d", capacity);
    }
    if (inject(t, "EnsureLocalCapacity")) {
        return JNI_ERR;
    }
    Frame & f = frame(t, "EnsureLocalCapacity");
    f.ensured = std::max(f.ensured, f.native_live + (size_t) capacity);
    return JNI_OK;
}

void fn_DeleteLocalRef(JNIEnv * env, jobject ref) {
    ThreadEnv & t = enter(env, "DeleteLocalRef", true);
    if (ref == nullptr) {
        return;
    }
    auto * r = reinterpret_cast<Ref *>(ref);
    Frame & f = frame(t, "DeleteLocalRef");
    if (!r->live) {
        fail("JNI DeleteLocalRef: the local reference %p is deleted already, or its call ended", (void *) ref);
    }
    if (r->serial != f.serial) {
        fail("JNI DeleteLocalRef: the local reference %p belongs to a different call", (void *) ref);
    }
    r->live = false;
    if (r->native) {
        f.native_live -= 1;
    }
}

jclass fn_FindClass(JNIEnv * env, const char * name) {
    ThreadEnv & t = enter(env, "FindClass");
    if (name == nullptr) {
        fail("JNI FindClass: null name");
    }
    if (inject(t, "FindClass")) {
        return nullptr;
    }
    return reinterpret_cast<jclass>(make_local(t, class_object(name), true));
}

jint fn_ThrowNew(JNIEnv * env, jclass cls, const char * msg) {
    ThreadEnv & t = enter(env, "ThrowNew");
    Obj * c = deref_kind(t, cls, "ThrowNew", Kind::Class);
    throw_exception(env, c->class_name, msg != nullptr ? msg : "");
    return JNI_OK;
}

jthrowable fn_ExceptionOccurred(JNIEnv * env) {
    ThreadEnv & t = enter(env, "ExceptionOccurred", true);
    return reinterpret_cast<jthrowable>(make_local(t, t.pending, true));
}

void fn_ExceptionClear(JNIEnv * env) {
    ThreadEnv & t = enter(env, "ExceptionClear", true);
    t.pending = nullptr;
}

jboolean fn_ExceptionCheck(JNIEnv * env) {
    ThreadEnv & t = enter(env, "ExceptionCheck", true);
    return t.pending != nullptr ? JNI_TRUE : JNI_FALSE;
}

jclass fn_GetObjectClass(JNIEnv * env, jobject obj) {
    ThreadEnv & t = enter(env, "GetObjectClass");
    Obj * o = deref(t, obj, "GetObjectClass", false);
    return reinterpret_cast<jclass>(make_local(t, class_object(o->class_name), true));
}

jmethodID lookup(ThreadEnv & t, jclass cls, const char * name, const char * sig, bool is_static, const char * fn) {
    Obj * c = deref_kind(t, cls, fn, Kind::Class);
    if (name == nullptr || sig == nullptr) {
        fail("JNI %s: null name or signature", fn);
    }
    if (inject(t, fn)) {
        return nullptr;
    }
    const std::string key = std::string(is_static ? "static|" : "method|") + c->class_name + "|" + name + "|" + sig;
    MethodEntry * found = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_methods.find(key);
        if (it != g_methods.end()) {
            found = it->second.get();
        } else if (!is_static && std::string(name) == "toString" && std::string(sig) == "()Ljava/lang/String;") {
            // Every object has toString. The fake gives the class name and the message.
            static MethodEntry to_string{"*", "toString", "()Ljava/lang/String;", nullptr, false};
            found = &to_string;
        }
    }
    if (found == nullptr) {
        throw_exception(&t, "java/lang/NoSuchMethodError", std::string(name) + sig);
        return nullptr;
    }
    return reinterpret_cast<jmethodID>(found);
}

jmethodID fn_GetMethodID(JNIEnv * env, jclass cls, const char * name, const char * sig) {
    ThreadEnv & t = enter(env, "GetMethodID");
    return lookup(t, cls, name, sig, false, "GetMethodID");
}

jmethodID fn_GetStaticMethodID(JNIEnv * env, jclass cls, const char * name, const char * sig) {
    ThreadEnv & t = enter(env, "GetStaticMethodID");
    return lookup(t, cls, name, sig, true, "GetStaticMethodID");
}

jobject fn_CallObjectMethodV(JNIEnv * env, jobject obj, jmethodID mid, va_list args) {
    ThreadEnv & t = enter(env, "CallObjectMethodV");
    Obj * o = deref(t, obj, "CallObjectMethodV", false);
    if (mid == nullptr) {
        fail("JNI CallObjectMethodV: null method");
    }
    auto * m = reinterpret_cast<MethodEntry *>(mid);
    if (m->is_static) {
        fail("JNI CallObjectMethodV: a static method");
    }
    if (inject(t, "CallObjectMethodV")) {
        return nullptr;
    }
    if (m->name == "toString") {
        std::string text = o->class_name;
        for (char & ch : text) {
            ch = ch == '/' ? '.' : ch;
        }
        if (o->kind == Kind::Throwable) {
            text += ": " + o->message;
        }
        return make_local(t, new_string_utf8(text), true);
    }
    return m->fn(env, args);
}

jobject fn_CallStaticObjectMethodV(JNIEnv * env, jclass cls, jmethodID mid, va_list args) {
    ThreadEnv & t = enter(env, "CallStaticObjectMethodV");
    deref_kind(t, cls, "CallStaticObjectMethodV", Kind::Class);
    if (mid == nullptr) {
        fail("JNI CallStaticObjectMethodV: null method");
    }
    auto * m = reinterpret_cast<MethodEntry *>(mid);
    if (!m->is_static) {
        fail("JNI CallStaticObjectMethodV: not a static method");
    }
    if (inject(t, "CallStaticObjectMethodV")) {
        return nullptr;
    }
    const Obj * before = t.pending;
    jobject out = m->fn(env, args);
    if (t.pending != before && out != nullptr) {
        fail("fake VM: the static method %s returned a value with a pending exception", m->name.c_str());
    }
    return out;
}

jstring fn_NewStringUTF(JNIEnv * env, const char * utf) {
    ThreadEnv & t = enter(env, "NewStringUTF");
    if (utf == nullptr) {
        return nullptr;
    }
    std::u16string chars;
    std::string why;
    if (!from_modified_utf8(utf, chars, why)) {
        fail("JNI NewStringUTF: the input is not valid modified UTF-8: %s (CheckJNI aborts the app here)", why.c_str());
    }
    if (inject(t, "NewStringUTF")) {
        return nullptr;
    }
    return reinterpret_cast<jstring>(make_local(t, new_string(chars), true));
}

const char * fn_GetStringUTFChars(JNIEnv * env, jstring s, jboolean * is_copy) {
    ThreadEnv & t = enter(env, "GetStringUTFChars");
    Obj * o = deref_kind(t, s, "GetStringUTFChars", Kind::String);
    if (inject(t, "GetStringUTFChars")) {
        return nullptr;
    }
    const std::string utf = to_modified_utf8(o->chars);
    auto * buf = static_cast<char *>(malloc(utf.size() + 1));
    memcpy(buf, utf.c_str(), utf.size() + 1);
    frame(t, "GetStringUTFChars").pins.push_back(Pin{buf, o, true});
    if (is_copy != nullptr) {
        *is_copy = JNI_TRUE;
    }
    return buf;
}

void fn_ReleaseStringUTFChars(JNIEnv * env, jstring s, const char * chars) {
    ThreadEnv & t = enter(env, "ReleaseStringUTFChars", true);
    Obj * o = deref_kind(t, s, "ReleaseStringUTFChars", Kind::String);
    auto & pins = frame(t, "ReleaseStringUTFChars").pins;
    for (auto it = pins.begin(); it != pins.end(); ++it) {
        if (it->ptr == chars && it->is_utf) {
            if (it->obj != o) {
                fail("JNI ReleaseStringUTFChars: the characters belong to a different string");
            }
            free(it->ptr);
            pins.erase(it);
            return;
        }
    }
    fail("JNI ReleaseStringUTFChars: the pointer %p is not pinned by this call", (const void *) chars);
}

jsize fn_GetArrayLength(JNIEnv * env, jarray array) {
    ThreadEnv & t = enter(env, "GetArrayLength");
    Obj * o = deref(t, array, "GetArrayLength", false);
    switch (o->kind) {
        case Kind::ByteArray:   return (jsize) o->bytes.size();
        case Kind::IntArray:    return (jsize) o->ints.size();
        case Kind::ObjectArray: return (jsize) o->elems.size();
        default: fail("JNI GetArrayLength: not an array (%s)", o->class_name.c_str());
    }
}

jobject fn_GetObjectArrayElement(JNIEnv * env, jobjectArray array, jsize index) {
    ThreadEnv & t = enter(env, "GetObjectArrayElement");
    Obj * o = deref_kind(t, array, "GetObjectArrayElement", Kind::ObjectArray);
    if (index < 0 || (size_t) index >= o->elems.size()) {
        throw_exception(env, "java/lang/ArrayIndexOutOfBoundsException", "index " + std::to_string(index));
        return nullptr;
    }
    return make_local(t, o->elems[(size_t) index], true);
}

jbyteArray fn_NewByteArray(JNIEnv * env, jsize length) {
    ThreadEnv & t = enter(env, "NewByteArray");
    if (length < 0) {
        fail("JNI NewByteArray: negative length %d", length);
    }
    if (inject(t, "NewByteArray")) {
        return nullptr;
    }
    auto obj = std::make_unique<Obj>();
    obj->kind       = Kind::ByteArray;
    obj->class_name = "[B";
    obj->bytes.assign((size_t) length, 0);
    return reinterpret_cast<jbyteArray>(make_local(t, own(std::move(obj)), true));
}

jintArray fn_NewIntArray(JNIEnv * env, jsize length) {
    ThreadEnv & t = enter(env, "NewIntArray");
    if (length < 0) {
        fail("JNI NewIntArray: negative length %d", length);
    }
    if (inject(t, "NewIntArray")) {
        return nullptr;
    }
    auto obj = std::make_unique<Obj>();
    obj->kind       = Kind::IntArray;
    obj->class_name = "[I";
    obj->ints.assign((size_t) length, 0);
    return reinterpret_cast<jintArray>(make_local(t, own(std::move(obj)), true));
}

jbyte * fn_GetByteArrayElements(JNIEnv * env, jbyteArray array, jboolean * is_copy) {
    ThreadEnv & t = enter(env, "GetByteArrayElements");
    Obj * o = deref_kind(t, array, "GetByteArrayElements", Kind::ByteArray);
    if (inject(t, "GetByteArrayElements")) {
        return nullptr;
    }
    // A copy with no spare byte, thus a read past the end is an ASan error.
    auto * buf = static_cast<jbyte *>(malloc(o->bytes.empty() ? 1 : o->bytes.size()));
    if (!o->bytes.empty()) {
        memcpy(buf, o->bytes.data(), o->bytes.size());
    }
    frame(t, "GetByteArrayElements").pins.push_back(Pin{buf, o, false});
    if (is_copy != nullptr) {
        *is_copy = JNI_TRUE;
    }
    return buf;
}

void fn_ReleaseByteArrayElements(JNIEnv * env, jbyteArray array, jbyte * elems, jint mode) {
    ThreadEnv & t = enter(env, "ReleaseByteArrayElements", true);
    Obj * o = deref_kind(t, array, "ReleaseByteArrayElements", Kind::ByteArray);
    auto & pins = frame(t, "ReleaseByteArrayElements").pins;
    for (auto it = pins.begin(); it != pins.end(); ++it) {
        if (it->ptr == elems && !it->is_utf) {
            if (it->obj != o) {
                fail("JNI ReleaseByteArrayElements: the elements belong to a different array");
            }
            if (mode == 0 || mode == JNI_COMMIT) {
                memcpy(o->bytes.data(), elems, o->bytes.size());
            }
            if (mode != JNI_COMMIT) {
                free(it->ptr);
                pins.erase(it);
            }
            return;
        }
    }
    fail("JNI ReleaseByteArrayElements: the pointer %p is not pinned by this call", (void *) elems);
}

void fn_GetIntArrayRegion(JNIEnv * env, jintArray array, jsize start, jsize len, jint * buf) {
    ThreadEnv & t = enter(env, "GetIntArrayRegion");
    Obj * o = deref_kind(t, array, "GetIntArrayRegion", Kind::IntArray);
    if (start < 0 || len < 0 || (size_t) start + (size_t) len > o->ints.size()) {
        throw_exception(env, "java/lang/ArrayIndexOutOfBoundsException", "region");
        return;
    }
    if (len > 0 && buf == nullptr) {
        fail("JNI GetIntArrayRegion: null buffer");
    }
    for (jsize i = 0; i < len; ++i) {
        buf[i] = o->ints[(size_t) (start + i)];
    }
}

void fn_SetByteArrayRegion(JNIEnv * env, jbyteArray array, jsize start, jsize len, const jbyte * buf) {
    ThreadEnv & t = enter(env, "SetByteArrayRegion");
    Obj * o = deref_kind(t, array, "SetByteArrayRegion", Kind::ByteArray);
    if (start < 0 || len < 0 || (size_t) start + (size_t) len > o->bytes.size()) {
        throw_exception(env, "java/lang/ArrayIndexOutOfBoundsException", "region");
        return;
    }
    if (len > 0 && buf == nullptr) {
        fail("JNI SetByteArrayRegion: null buffer");
    }
    if (len > 0) {
        memcpy(o->bytes.data() + start, buf, (size_t) len);
    }
}

void * fn_GetDirectBufferAddress(JNIEnv * env, jobject buffer) {
    ThreadEnv & t = enter(env, "GetDirectBufferAddress");
    Obj * o = deref(t, buffer, "GetDirectBufferAddress", false);
    if (o->kind != Kind::DirectBuffer) {
        return nullptr;
    }
    return o->bytes.empty() ? nullptr : o->bytes.data();
}

jlong fn_GetDirectBufferCapacity(JNIEnv * env, jobject buffer) {
    ThreadEnv & t = enter(env, "GetDirectBufferCapacity");
    Obj * o = deref(t, buffer, "GetDirectBufferCapacity", false);
    return o->kind == Kind::DirectBuffer ? (jlong) o->bytes.size() : -1;
}

/** Every slot that the table does not implement points here. */
void fn_unexpected() {
    fail("JNI: the native code called a JNI function that the fake VM does not implement");
}

const Table & table() {
    static const Table tbl = [] {
        Table t;
        // Each slot of the table is a pointer. The ones that follow replace the trap.
        auto ** slots = reinterpret_cast<void **>(&t);
        for (size_t i = 0; i < sizeof(Table) / sizeof(void *); ++i) {
            slots[i] = reinterpret_cast<void *>(&fn_unexpected);
        }
        t.EnsureLocalCapacity       = fn_EnsureLocalCapacity;
        t.DeleteLocalRef            = fn_DeleteLocalRef;
        t.FindClass                 = fn_FindClass;
        t.ThrowNew                  = fn_ThrowNew;
        t.ExceptionOccurred         = fn_ExceptionOccurred;
        t.ExceptionClear            = fn_ExceptionClear;
        t.ExceptionCheck            = fn_ExceptionCheck;
        t.GetObjectClass            = fn_GetObjectClass;
        t.GetMethodID               = fn_GetMethodID;
        t.GetStaticMethodID         = fn_GetStaticMethodID;
        t.CallObjectMethodV         = fn_CallObjectMethodV;
        t.CallStaticObjectMethodV   = fn_CallStaticObjectMethodV;
        t.NewStringUTF              = fn_NewStringUTF;
        t.GetStringUTFChars         = fn_GetStringUTFChars;
        t.ReleaseStringUTFChars     = fn_ReleaseStringUTFChars;
        t.GetArrayLength            = fn_GetArrayLength;
        t.GetObjectArrayElement     = fn_GetObjectArrayElement;
        t.NewByteArray              = fn_NewByteArray;
        t.NewIntArray               = fn_NewIntArray;
        t.GetByteArrayElements      = fn_GetByteArrayElements;
        t.ReleaseByteArrayElements  = fn_ReleaseByteArrayElements;
        t.GetIntArrayRegion         = fn_GetIntArrayRegion;
        t.SetByteArrayRegion        = fn_SetByteArrayRegion;
        t.GetDirectBufferAddress    = fn_GetDirectBufferAddress;
        t.GetDirectBufferCapacity   = fn_GetDirectBufferCapacity;
        return t;
    }();
    return tbl;
}

}  // namespace

void fail(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "\n==FAKEJNI== ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    if (__sanitizer_print_stack_trace != nullptr) {
        __sanitizer_print_stack_trace();
    }
    crash_input::stop();
}

bool relaxed(const char * check) {
    static const std::set<std::string> names = [] {
        std::set<std::string> out;
        const char * v = getenv("FAKEJNI_RELAX");
        std::string s = v != nullptr ? v : "";
        size_t pos = 0;
        while (pos <= s.size()) {
            const size_t end = s.find(',', pos);
            const std::string item = s.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
            if (!item.empty()) {
                out.insert(item);
            }
            if (end == std::string::npos) {
                break;
            }
            pos = end + 1;
        }
        return out;
    }();
    return names.count(check) != 0;
}

JNIEnv * env() {
    thread_local std::unique_ptr<ThreadEnv> t = [] {
        auto e = std::make_unique<ThreadEnv>();
        e->functions = &table();
        e->owner     = std::this_thread::get_id();
        return e;
    }();
    return t.get();
}

std::string to_modified_utf8(const std::u16string & chars) {
    // The conversion of ART (ConvertUtf16ToModifiedUtf8): a valid surrogate pair
    // becomes 4 bytes, U+0000 becomes C0 80, and a lone surrogate becomes 3 bytes.
    std::string out;
    for (size_t i = 0; i < chars.size(); ++i) {
        const uint32_t ch = chars[i];
        if (ch > 0 && ch <= 0x7F) {
            out.push_back((char) ch);
            continue;
        }
        if (ch >= 0xD800 && ch <= 0xDBFF && i + 1 < chars.size()) {
            const uint32_t ch2 = chars[i + 1];
            if (ch2 >= 0xDC00 && ch2 <= 0xDFFF) {
                const uint32_t cp = ((ch - 0xD800) << 10) + (ch2 - 0xDC00) + 0x10000;
                out.push_back((char) (0xF0 | (cp >> 18)));
                out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
                out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
                out.push_back((char) (0x80 | (cp & 0x3F)));
                ++i;
                continue;
            }
        }
        if (ch <= 0x7FF) {
            out.push_back((char) (0xC0 | (ch >> 6)));
            out.push_back((char) (0x80 | (ch & 0x3F)));
        } else {
            out.push_back((char) (0xE0 | (ch >> 12)));
            out.push_back((char) (0x80 | ((ch >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (ch & 0x3F)));
        }
    }
    return out;
}

Obj * new_string(const std::u16string & chars) {
    auto obj = std::make_unique<Obj>();
    obj->kind       = Kind::String;
    obj->class_name = "java/lang/String";
    obj->chars      = chars;
    return own(std::move(obj));
}

Obj * new_string_utf8(const std::string & utf8) {
    std::u16string chars;
    const auto * p = reinterpret_cast<const unsigned char *>(utf8.data());
    const size_t n = utf8.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char c = p[i];
        int extra = c < 0x80 ? 0 : (c & 0xE0) == 0xC0 ? 1 : (c & 0xF0) == 0xE0 ? 2 : (c & 0xF8) == 0xF0 ? 3 : -1;
        uint32_t cp = extra == 0 ? c : extra == 1 ? (c & 0x1F) : extra == 2 ? (c & 0x0F) : (c & 0x07);
        bool ok = extra >= 0 && i + (size_t) extra < n;
        for (int k = 1; ok && k <= extra; ++k) {
            ok = (p[i + (size_t) k] & 0xC0) == 0x80;
            cp = (cp << 6) | (p[i + (size_t) k] & 0x3F);
        }
        if (!ok) {
            chars.push_back(u'�');
            i += 1;
            continue;
        }
        i += 1 + (size_t) extra;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            chars.push_back((char16_t) (0xD800 + (cp >> 10)));
            chars.push_back((char16_t) (0xDC00 + (cp & 0x3FF)));
        } else {
            chars.push_back((char16_t) cp);
        }
    }
    return new_string(chars);
}

Obj * new_byte_array(const uint8_t * data, size_t n) {
    auto obj = std::make_unique<Obj>();
    obj->kind       = Kind::ByteArray;
    obj->class_name = "[B";
    obj->bytes.assign(reinterpret_cast<const int8_t *>(data), reinterpret_cast<const int8_t *>(data) + n);
    return own(std::move(obj));
}

Obj * new_object_array(const std::string & element_class, std::vector<Obj *> elems) {
    auto obj = std::make_unique<Obj>();
    obj->kind       = Kind::ObjectArray;
    obj->class_name = "[L" + element_class + ";";
    obj->elems      = std::move(elems);
    return own(std::move(obj));
}

Obj * new_direct_buffer(const uint8_t * data, size_t n) {
    auto obj = std::make_unique<Obj>();
    obj->kind       = Kind::DirectBuffer;
    obj->class_name = "java/nio/DirectByteBuffer";
    obj->bytes.assign(reinterpret_cast<const int8_t *>(data), reinterpret_cast<const int8_t *>(data) + n);
    return own(std::move(obj));
}

Obj * class_object(const std::string & name) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_classes.find(name);
        if (it != g_classes.end()) {
            return it->second;
        }
    }
    auto obj = std::make_unique<Obj>();
    obj->kind       = Kind::Class;
    obj->class_name = name;
    Obj * raw = own(std::move(obj));
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_classes.emplace(name, raw).first->second;
}

Obj * object_of(JNIEnv * env, jobject ref, const char * what) {
    return deref(state(env, what), ref, what, true);
}

jobject new_local(JNIEnv * env, Obj * obj, bool counts_for_native) {
    return make_local(state(env, "new_local"), obj, counts_for_native);
}

void throw_exception(JNIEnv * env, const std::string & class_name, const std::string & message) {
    ThreadEnv & t = state(env, "throw");
    auto obj = std::make_unique<Obj>();
    obj->kind       = Kind::Throwable;
    obj->class_name = class_name;
    obj->message    = message;
    t.pending = own(std::move(obj));
}

void begin_call(JNIEnv * env, const char * name) {
    ThreadEnv & t = state(env, name);
    if (t.pending != nullptr) {
        fail("fake VM: %s starts with a pending exception", name);
    }
    Frame f;
    f.name = name;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        f.serial = ++g_serial;
    }
    t.frames.push_back(std::move(f));
}

jobject arg(JNIEnv * env, Obj * obj) {
    return make_local(state(env, "arg"), obj, false);
}

CallOutcome end_call(JNIEnv * env) {
    ThreadEnv & t = state(env, "end_call");
    Frame & f = frame(t, "end_call");
    if (!f.pins.empty()) {
        fail("JNI %s: the call returned with %zu pinned buffers that it did not release (%s)", f.name, f.pins.size(),
             f.pins.front().is_utf ? "GetStringUTFChars" : "GetByteArrayElements");
    }
    CallOutcome out;
    out.peak_local_refs = f.peak;
    for (Ref * r : f.refs) {
        r->live = false;
    }
    t.frames.pop_back();
    if (t.pending != nullptr) {
        out.threw             = true;
        out.exception_class   = t.pending->class_name;
        out.exception_message = t.pending->message;
        t.pending             = nullptr;
    }
    return out;
}

void register_static(const std::string & class_name, const std::string & name, const std::string & sig,
                     StaticMethod method) {
    auto entry = std::make_unique<MethodEntry>();
    entry->class_name = class_name;
    entry->name       = name;
    entry->sig        = sig;
    entry->fn         = std::move(method);
    entry->is_static  = true;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_methods["static|" + class_name + "|" + name + "|" + sig] = std::move(entry);
}

void clear_statics() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto it = g_methods.begin(); it != g_methods.end();) {
        it = it->first.rfind("static|", 0) == 0 ? g_methods.erase(it) : std::next(it);
    }
}

void arm_alloc_failure(JNIEnv * env, int countdown) {
    state(env, "arm_alloc_failure").countdown = countdown;
}

uint64_t injected_faults(JNIEnv * env) {
    return state(env, "injected_faults").injected;
}

void reset() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_refs.clear();
    g_classes.clear();
    g_heap.clear();
}

}  // namespace fakejni
