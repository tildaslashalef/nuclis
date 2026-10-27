// Audio decoding through AudioToolbox, behind a C-compatible interface: any
// format the platform reads (WAV, AIFF, CAF, MP3, M4A/AAC, FLAC, ALAC)
// becomes 16 kHz mono float samples. ExtAudioFile converts the sample rate
// (at its best quality) and the sample format; channels are averaged here.
// Whole: the caller copies the samples out and frees them with
// nu_audio_free. In pieces: nu_audio_open, nu_audio_read until it gives
// none, nu_audio_close; the bytes outlive the reader.
#include <AudioToolbox/AudioToolbox.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t * bytes;
    size_t len;
} Source;

static OSStatus read_bytes(void * client, SInt64 position, UInt32 count, void * buffer, UInt32 * actual) {
    const Source * s = client;
    if (position < 0) return kAudioFileInvalidPacketOffsetError;
    if ((uint64_t) position >= s->len) {
        *actual = 0;
        return noErr;
    }
    size_t n = s->len - (size_t) position;
    if (n > count) n = count;
    memcpy(buffer, s->bytes + position, n);
    *actual = (UInt32) n;
    return noErr;
}

static SInt64 source_size(void * client) {
    return (SInt64) ((const Source *) client)->len;
}

// Opens `source` as an ExtAudioFile delivering 16 kHz float frames of the
// file's own channels; `total` is the length at 16 kHz. Returns 0, or 1 when
// the bytes are not decodable audio (nothing is left open).
static int open_ext(Source * source, AudioFileID * file, ExtAudioFileRef * ext, UInt32 * channels, uint64_t * total) {
    *file = NULL;
    *ext = NULL;
    if (AudioFileOpenWithCallbacks(source, read_bytes, NULL, source_size, NULL, 0, file) != noErr) return 1;
    if (ExtAudioFileWrapAudioFileID(*file, false, ext) != noErr) {
        AudioFileClose(*file);
        return 1;
    }
    AudioStreamBasicDescription in;
    UInt32 size = sizeof(in);
    SInt64 frames = 0;
    UInt32 frames_size = sizeof(frames);
    if (ExtAudioFileGetProperty(*ext, kExtAudioFileProperty_FileDataFormat, &size, &in) != noErr || in.mSampleRate <= 0 || in.mChannelsPerFrame == 0 || in.mChannelsPerFrame > 64) goto fail;
    if (ExtAudioFileGetProperty(*ext, kExtAudioFileProperty_FileLengthFrames, &frames_size, &frames) != noErr || frames < 0) goto fail;
    {
        *channels = in.mChannelsPerFrame;
        AudioStreamBasicDescription client = {0};
        client.mSampleRate = 16000;
        client.mFormatID = kAudioFormatLinearPCM;
        client.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
        client.mBitsPerChannel = 32;
        client.mChannelsPerFrame = *channels;
        client.mFramesPerPacket = 1;
        client.mBytesPerFrame = 4 * *channels;
        client.mBytesPerPacket = 4 * *channels;
        if (ExtAudioFileSetProperty(*ext, kExtAudioFileProperty_ClientDataFormat, sizeof(client), &client) != noErr) goto fail;
        if (in.mSampleRate != 16000) {
            AudioConverterRef converter = NULL;
            UInt32 converter_size = sizeof(converter);
            if (ExtAudioFileGetProperty(*ext, kExtAudioFileProperty_AudioConverter, &converter_size, &converter) == noErr && converter) {
                UInt32 quality = kAudioConverterQuality_Max;
                UInt32 complexity = kAudioConverterSampleRateConverterComplexity_Mastering;
                AudioConverterSetProperty(converter, kAudioConverterSampleRateConverterQuality, sizeof(quality), &quality);
                AudioConverterSetProperty(converter, kAudioConverterSampleRateConverterComplexity, sizeof(complexity), &complexity);
                CFArrayRef config = NULL;
                ExtAudioFileSetProperty(*ext, kExtAudioFileProperty_ConverterConfig, sizeof(config), &config);
            }
        }
        *total = (uint64_t) ((double) frames * 16000.0 / in.mSampleRate + 0.5);
        return 0;
    }
fail:
    ExtAudioFileDispose(*ext);
    AudioFileClose(*file);
    return 1;
}

// Returns 0 with `samples` (malloc'd, `count` of them, at most `max_samples`)
// and `total`, the clip's length at 16 kHz before the bound, on success; 1
// when the bytes are not decodable audio; 3 on an allocation failure.
int nu_audio_decode(const uint8_t * bytes, size_t len, uint32_t max_samples, float ** samples, uint32_t * count, uint64_t * total) {
    Source source = {bytes, len};
    AudioFileID file = NULL;
    ExtAudioFileRef ext = NULL;
    UInt32 channels = 0;
    if (open_ext(&source, &file, &ext, &channels, total) != 0) return 1;
    int result = 1;
    {
        uint64_t capacity = *total < max_samples ? *total : max_samples;
        // The estimate may be a frame short after conversion; leave room.
        capacity += 64;
        if (capacity > (uint64_t) max_samples + 64) capacity = (uint64_t) max_samples + 64;
        float * mono = malloc((size_t) capacity * sizeof(float));
        float * chunk = malloc((size_t) 4096 * channels * sizeof(float));
        if (!mono || !chunk) {
            free(mono);
            free(chunk);
            result = 3;
            goto done;
        }
        uint32_t written = 0;
        while (written < max_samples) {
            UInt32 want = 4096;
            AudioBufferList list;
            list.mNumberBuffers = 1;
            list.mBuffers[0].mNumberChannels = channels;
            list.mBuffers[0].mDataByteSize = want * 4 * channels;
            list.mBuffers[0].mData = chunk;
            if (ExtAudioFileRead(ext, &want, &list) != noErr) {
                free(mono);
                free(chunk);
                goto done;
            }
            if (want == 0) break;
            for (UInt32 i = 0; i < want && written < max_samples && written < capacity; i++) {
                float sum = 0;
                for (UInt32 c = 0; c < channels; c++) sum += chunk[i * channels + c];
                mono[written++] = channels == 1 ? sum : sum / (float) channels;
            }
            if (written >= capacity) break;
        }
        free(chunk);
        if (written > *total) *total = written;
        *samples = mono;
        *count = written;
        result = 0;
    }
done:
    ExtAudioFileDispose(ext);
    AudioFileClose(file);
    return result;
}

void nu_audio_free(float * samples) { free(samples); }

// A decoding in pieces: the frames `nu_audio_decode` reads, 4096 at a time,
// handed out as asked. `bytes` must outlive it.
typedef struct {
    Source source;
    AudioFileID file;
    ExtAudioFileRef ext;
    UInt32 channels;
    float * chunk;
    UInt32 have;
    UInt32 at;
} Reader;

// Returns 0 with `reader` and `total` (the length at 16 kHz, an estimate
// the reads may pass by a frame); 1 when the bytes are not decodable audio;
// 3 on an allocation failure.
int nu_audio_open(const uint8_t * bytes, size_t len, void ** reader, uint64_t * total) {
    Reader * r = calloc(1, sizeof(Reader));
    if (!r) return 3;
    r->source = (Source) {bytes, len};
    if (open_ext(&r->source, &r->file, &r->ext, &r->channels, total) != 0) {
        free(r);
        return 1;
    }
    r->chunk = malloc((size_t) 4096 * r->channels * sizeof(float));
    if (!r->chunk) {
        ExtAudioFileDispose(r->ext);
        AudioFileClose(r->file);
        free(r);
        return 3;
    }
    *reader = r;
    return 0;
}

// Writes up to `max` mono samples to `out`; `count` 0 is the end. Returns
// 0, or 1 when the file fails to decode midway.
int nu_audio_read(void * reader, float * out, uint32_t max, uint32_t * count) {
    Reader * r = reader;
    uint32_t written = 0;
    while (written < max) {
        if (r->at == r->have) {
            UInt32 want = 4096;
            AudioBufferList list;
            list.mNumberBuffers = 1;
            list.mBuffers[0].mNumberChannels = r->channels;
            list.mBuffers[0].mDataByteSize = want * 4 * r->channels;
            list.mBuffers[0].mData = r->chunk;
            if (ExtAudioFileRead(r->ext, &want, &list) != noErr) return 1;
            r->have = want;
            r->at = 0;
            if (want == 0) break;
        }
        for (; r->at < r->have && written < max; r->at++) {
            float sum = 0;
            for (UInt32 c = 0; c < r->channels; c++) sum += r->chunk[r->at * r->channels + c];
            out[written++] = r->channels == 1 ? sum : sum / (float) r->channels;
        }
    }
    *count = written;
    return 0;
}

void nu_audio_close(void * reader) {
    Reader * r = reader;
    ExtAudioFileDispose(r->ext);
    AudioFileClose(r->file);
    free(r->chunk);
    free(r);
}
