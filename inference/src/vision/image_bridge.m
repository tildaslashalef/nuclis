// Image decoding through ImageIO, behind a C-compatible interface: any format
// the platform reads (PNG, JPEG, HEIC, WebP, TIFF, GIF, BMP) becomes tightly
// packed 8-bit RGB rows. The caller copies the pixels out and frees them
// with nu_image_free; nothing here outlives the call.
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// The byte offset of red in a pixel of an 8-bit RGB image whose stored
// values are straight (not premultiplied), or SIZE_MAX for any other layout.
static size_t direct_offset(CGImageRef image) {
    if (CGImageGetBitsPerComponent(image) != 8) return SIZE_MAX;
    if (CGColorSpaceGetModel(CGImageGetColorSpace(image)) != kCGColorSpaceModelRGB) return SIZE_MAX;
    CGBitmapInfo order = CGImageGetBitmapInfo(image) & kCGBitmapByteOrderMask;
    if (order != kCGBitmapByteOrderDefault && order != kCGBitmapByteOrder32Big) return SIZE_MAX;
    if (CGImageGetBitmapInfo(image) & kCGBitmapFloatComponents) return SIZE_MAX;
    size_t bpp = CGImageGetBitsPerPixel(image);
    switch (CGImageGetAlphaInfo(image)) {
        case kCGImageAlphaNone: return bpp == 24 ? 0 : SIZE_MAX;
        case kCGImageAlphaLast:
        case kCGImageAlphaNoneSkipLast: return bpp == 32 ? 0 : SIZE_MAX;
        case kCGImageAlphaFirst:
        case kCGImageAlphaNoneSkipFirst: return bpp == 32 ? 1 : SIZE_MAX;
        default: return SIZE_MAX;
    }
}

static int copy_direct(CGImageRef image, size_t w, size_t h, size_t red, uint32_t * width, uint32_t * height, uint8_t ** pixels) {
    CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(image));
    if (!data) return 3;
    size_t stride = CGImageGetBytesPerRow(image);
    size_t bytes_per_pixel = CGImageGetBitsPerPixel(image) / 8;
    int result = 3;
    if ((size_t) CFDataGetLength(data) < stride * (h - 1) + w * bytes_per_pixel) {
        result = 1;
    } else {
        const uint8_t * source = CFDataGetBytePtr(data);
        uint8_t * rgb = malloc(w * h * 3);
        if (rgb) {
            for (size_t y = 0; y < h; y++) {
                const uint8_t * row = source + y * stride + red;
                for (size_t x = 0; x < w; x++) memcpy(rgb + (y * w + x) * 3, row + x * bytes_per_pixel, 3);
            }
            *width = (uint32_t) w;
            *height = (uint32_t) h;
            *pixels = rgb;
            result = 0;
        }
    }
    CFRelease(data);
    return result;
}

// Returns 0 with `width`, `height`, and `pixels` (width * height * 3 bytes,
// malloc'd) on success; 1 when the bytes are not a decodable image; 2 when
// the decoded size exceeds `max_pixels`; 3 on an allocation failure.
int nu_image_decode(const uint8_t * bytes, size_t len, uint32_t max_pixels, uint32_t * width, uint32_t * height, uint8_t ** pixels) {
    int result = 1;
    @autoreleasepool {
        CFDataRef data = CFDataCreateWithBytesNoCopy(NULL, bytes, (CFIndex) len, kCFAllocatorNull);
        if (!data) return 3;
        CGImageSourceRef source = CGImageSourceCreateWithData(data, NULL);
        CFRelease(data);
        if (!source) return 1;
        CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        CFRelease(source);
        if (!image) return 1;
        size_t w = CGImageGetWidth(image), h = CGImageGetHeight(image);
        if (w == 0 || h == 0 || w > UINT32_MAX || h > UINT32_MAX || (uint64_t) w * h > max_pixels) {
            CGImageRelease(image);
            return 2;
        }
        // The common 8-bit RGB layouts are copied as stored, alpha dropped
        // unpremultiplied and no colour conversion, as PIL's convert("RGB")
        // and stb_image do; drawing would premultiply a translucent pixel.
        size_t direct = direct_offset(image);
        if (direct != SIZE_MAX) {
            result = copy_direct(image, w, h, direct, width, height, pixels);
            CGImageRelease(image);
            return result;
        }
        // Anything else is drawn into an RGBA8 context so every colour
        // space and depth resolves to one layout, then alpha is dropped.
        uint8_t * rgba = calloc(w * h, 4);
        CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
        CGContextRef context = rgba && space ? CGBitmapContextCreate(rgba, w, h, 8, w * 4, space, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big) : NULL;
        if (context) {
            CGContextDrawImage(context, CGRectMake(0, 0, (CGFloat) w, (CGFloat) h), image);
            uint8_t * rgb = malloc(w * h * 3);
            if (rgb) {
                for (size_t i = 0; i < w * h; i++) {
                    rgb[i * 3] = rgba[i * 4];
                    rgb[i * 3 + 1] = rgba[i * 4 + 1];
                    rgb[i * 3 + 2] = rgba[i * 4 + 2];
                }
                *width = (uint32_t) w;
                *height = (uint32_t) h;
                *pixels = rgb;
                result = 0;
            } else result = 3;
            CGContextRelease(context);
        } else result = 3;
        if (space) CGColorSpaceRelease(space);
        free(rgba);
        CGImageRelease(image);
    }
    return result;
}

void nu_image_free(uint8_t * pixels) { free(pixels); }
