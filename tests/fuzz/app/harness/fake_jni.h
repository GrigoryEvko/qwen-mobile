/**
 * A fake Java VM for the fuzz harnesses of the JNI layer.
 *
 * The harness calls the JNI entry points of llama_jni.cpp directly, with a
 * JNIEnv whose function table comes from this file. The table implements
 * the small part of JNI that the app uses, and it does the checks that the
 * CheckJNI mode of ART does on a debuggable build (the release build of the
 * app is debuggable, thus ART aborts the app on each of these errors):
 *
 * - A JNI call with a pending exception, other than the calls that JNI permits.
 * - A null argument where JNI requires an object, and an object of the wrong type.
 * - A local reference that was deleted, or that belongs to a call that ended.
 * - A local reference that the call deletes two times.
 * - Pinned string characters or array elements that the call does not release.
 * - Text for NewStringUTF that is not modified UTF-8.
 * - More live local references than the call ensured (JNI guarantees 16).
 * - A JNIEnv that a different thread uses.
 *
 * Each error goes to fail(), which writes the message and aborts. Thus
 * libFuzzer records the input. An allocation fault plan makes the allocating
 * calls return null with an OutOfMemoryError pending, as a full Java heap
 * does. Thus the fuzzer reaches every error path of the JNI layer.
 *
 * The objects live on a heap that reset() releases. The harness calls it
 * between iterations. A local reference stays readable after its call ends,
 * thus a later use of it is an error with a message and not a crash.
 *
 * The file compiles against the jni.h of the NDK and against the jni.h of a
 * JDK. The function table type comes from the functions member.
 */
#pragma once

#include <jni.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace fakejni {

/** The type of a heap object. */
enum class Kind : uint8_t { Class, String, ByteArray, IntArray, ObjectArray, DirectBuffer, Throwable };

/** One object of the fake Java heap. */
struct Obj {
    Kind                 kind = Kind::Class;
    /** The class name in JNI form, for example "java/lang/String". */
    std::string          class_name;
    /** The UTF-16 characters of a String. */
    std::u16string       chars;
    /** The elements of a byte[] and the storage of a direct ByteBuffer. */
    std::vector<int8_t>  bytes;
    /** The elements of an int[]. */
    std::vector<int32_t> ints;
    /** The elements of an Object[]. An element can be null. */
    std::vector<Obj *>   elems;
    /** The message of a Throwable. */
    std::string          message;
};

/** What the Java side sees after a native call returns. */
struct CallOutcome {
    /** True when the call returned with a pending exception. */
    bool        threw = false;
    /** The class and the message of that exception. */
    std::string exception_class;
    std::string exception_message;
    /** The peak of the live local references that the native code made during the call. */
    size_t      peak_local_refs = 0;
};

/**
 * The implementation of a static Java method that native code calls. The
 * arguments come as a va_list, in the order of the signature. The function
 * returns a local reference, or null. It can set a pending exception with
 * throw_exception().
 */
using StaticMethod = std::function<jobject(JNIEnv * env, va_list args)>;

/** Write the message to stderr and abort the process. */
[[noreturn]] void fail(const char * fmt, ...) __attribute__((format(printf, 1, 2)));

/** The JNIEnv of the calling thread. The first call on a thread makes it. */
JNIEnv * env();

/** Make a String from UTF-16 characters. The heap owns the object. */
Obj * new_string(const std::u16string & chars);

/** Make a String from standard UTF-8. A byte that is not UTF-8 becomes U+FFFD. */
Obj * new_string_utf8(const std::string & utf8);

/** Make a byte[] with a copy of the bytes. */
Obj * new_byte_array(const uint8_t * data, size_t n);

/** Make an Object[] of the class with the elements. */
Obj * new_object_array(const std::string & element_class, std::vector<Obj *> elems);

/** Make a direct ByteBuffer with a copy of the bytes. Its capacity is the byte count. */
Obj * new_direct_buffer(const uint8_t * data, size_t n);

/** The Class object of a JNI class name. The heap keeps one object for each name. */
Obj * class_object(const std::string & name);

/** The object of a local reference. Fails when the reference is not live. */
Obj * object_of(JNIEnv * env, jobject ref, const char * what);

/** A new local reference to the object in the current call of the thread, or null for null. */
jobject new_local(JNIEnv * env, Obj * obj, bool counts_for_native);

/** Set a pending exception of the class with the message. */
void throw_exception(JNIEnv * env, const std::string & class_name, const std::string & message);

/** Start a native call on the thread: the arguments that follow are its local references. */
void begin_call(JNIEnv * env, const char * name);

/** A local reference for one argument of the call, or null for null. */
jobject arg(JNIEnv * env, Obj * obj);

/**
 * End the native call: do the checks of the end of a call, release its local
 * references, and take its pending exception.
 */
CallOutcome end_call(JNIEnv * env);

/** Register a static method of a class. The key is the class, the name and the signature. */
void register_static(const std::string & class_name, const std::string & name, const std::string & sig,
                     StaticMethod method);

/** Remove every static method. */
void clear_statics();

/**
 * The allocation fault plan of the calling thread: the countdown-th
 * allocating JNI call from now fails with an OutOfMemoryError, and the
 * calls after it succeed. A negative value turns the plan off.
 */
void arm_alloc_failure(JNIEnv * env, int countdown);

/** The number of allocation faults that the plan injected on this thread. */
uint64_t injected_faults(JNIEnv * env);

/** Release every heap object and every local reference. No call can be open. */
void reset();

/**
 * Turn one check off by its name: "capacity" (the local reference
 * capacity). Thus a fuzz run continues past a defect that it recorded.
 * The environment variable FAKEJNI_RELAX holds a comma-separated list of
 * names at the first call.
 */
bool relaxed(const char * check);

/** The modified UTF-8 of UTF-16 characters, as ART makes it for GetStringUTFChars. */
std::string to_modified_utf8(const std::u16string & chars);

}  // namespace fakejni
