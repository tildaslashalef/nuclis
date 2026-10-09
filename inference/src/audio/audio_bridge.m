// Audio decoding through AudioToolbox, behind a C-compatible interface: any
// format the platform reads (WAV, AIFF, CAF, MP3, M4A/AAC, FLAC, ALAC)
// becomes 16 kHz mono float samples. ExtAudioFile converts the sample rate
// (at its best quality) and the sample format; channels are averaged here.
// The caller copies the samples out and frees them with nu_audio_free;
// nothing here outlives the call.
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

// Returns 0 with `samples` (malloc'd, `count` of them, at most `max_samples`)
// and `total`, the clip's length at 16 kHz before the bound, on success; 1
// when the bytes are not decodable audio; 3 on an allocation failure.
int nu_audio_decode(const uint8_t * bytes, size_t len, uint32_t max_samples, float ** samples, uint32_t * count, uint64_t * total) {
    Source source = {bytes, len};
    AudioFileID file = NULL;
    if (AudioFileOpenWithCallbacks(&source, read_bytes, NULL, source_size, NULL, 0, &file) != noErr) return 1;
    ExtAudioFileRef ext = NULL;
    if (ExtAudioFileWrapAudioFileID(file, false, &ext) != noErr) {
        AudioFileClose(file);
        return 1;
    }
    int result = 1;
    AudioStreamBasicDescription in;
    UInt32 size = sizeof(in);
    SInt64 frames = 0;
    UInt32 frames_size = sizeof(frames);
    if (ExtAudioFileGetProperty(ext, kExtAudioFileProperty_FileDataFormat, &size, &in) != noErr || in.mSampleRate <= 0 || in.mChannelsPerFrame == 0 || in.mChannelsPerFrame > 64) goto done;
    if (ExtAudioFileGetProperty(ext, kExtAudioFileProperty_FileLengthFrames, &frames_size, &frames) != noErr || frames < 0) goto done;
    {
        const UInt32 channels = in.mChannelsPerFrame;
        AudioStreamBasicDescription client = {0};
        client.mSampleRate = 16000;
        client.mFormatID = kAudioFormatLinearPCM;
        client.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
        client.mBitsPerChannel = 32;
        client.mChannelsPerFrame = channels;
        client.mFramesPerPacket = 1;
        client.mBytesPerFrame = 4 * channels;
        client.mBytesPerPacket = 4 * channels;
        if (ExtAudioFileSetProperty(ext, kExtAudioFileProperty_ClientDataFormat, sizeof(client), &client) != noErr) goto done;
        if (in.mSampleRate != 16000) {
            AudioConverterRef converter = NULL;
            UInt32 converter_size = sizeof(converter);
            if (ExtAudioFileGetProperty(ext, kExtAudioFileProperty_AudioConverter, &converter_size, &converter) == noErr && converter) {
                UInt32 quality = kAudioConverterQuality_Max;
                UInt32 complexity = kAudioConverterSampleRateConverterComplexity_Mastering;
                AudioConverterSetProperty(converter, kAudioConverterSampleRateConverterQuality, sizeof(quality), &quality);
                AudioConverterSetProperty(converter, kAudioConverterSampleRateConverterComplexity, sizeof(complexity), &complexity);
                CFArrayRef config = NULL;
                ExtAudioFileSetProperty(ext, kExtAudioFileProperty_ConverterConfig, sizeof(config), &config);
            }
        }
        *total = (uint64_t) ((double) frames * 16000.0 / in.mSampleRate + 0.5);
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
