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
        // Draw into an RGBA8 context so every source colour space, depth,
        // alpha, and orientation resolves to one layout; alpha is then
        // dropped (the reference composites over nothing either).
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
