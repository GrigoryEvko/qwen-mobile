// The last lines of the copy of the mapping table of the DSP for fuzz_mmap (CMakeLists.txt writes the copy).
// The functions of htp-mmap-api.h, in the translation unit of the static prep_op_bufs.

size_t hexmmap_ctx_size(void) {
    return sizeof(struct htp_context);
}

uint32_t hexmmap_slots(void) {
    return HTP_MAX_MMAPS;
}

void hexmmap_init(void * ctx, uint64_t max_vmem) {
    memset(ctx, 0, sizeof(struct htp_context));
    ((struct htp_context *) ctx)->max_vmem = max_vmem;
}

void hexmmap_prep(void * ctx, struct htp_buf_desc * bufs, uint32_t n_bufs, uint32_t seq, bool lru) {
    prep_op_bufs((struct htp_context *) ctx, bufs, n_bufs, seq, lru);
}

void hexmmap_slot(const void * ctx, uint32_t i, uint32_t * fd, uint64_t * size, uint64_t * base) {
    const struct htp_mmap * m = &((const struct htp_context *) ctx)->mmap[i];
    *fd   = m->fd;
    *size = m->size;
    *base = m->base;
}
