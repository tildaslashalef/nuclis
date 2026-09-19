// Prints the pieces the pinned reference's pre-tokenizer produces for a
// regex, straight from its `unicode_regex_split` (token ids cannot show
// boundaries: BPE merges hide them). Input: strings separated by U+001E on
// stdin; output: pieces separated by U+001F, records by U+001E. Build against
// the reference checkout (docs/reference/reference-baseline.md):
//
//   R=.zig-cache/reference/llama.cpp
//   clang++ -std=c++17 -O1 -I $R/src -I $R/include -I $R/ggml/include \
//     scripts/reference-split.cpp $R/src/unicode.cpp $R/src/unicode-data.cpp \
//     -o .zig-cache/reference/split
//
// The default regex is the form the reference runs for the `llama4` label
// (docs/reference/muse-glimmer.md § Tokenizer); pass another as argv[1].
#include "unicode.h"
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string regex = argc > 1 ? argv[1] :
        "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
        "|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
        "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    std::string input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    size_t start = 0;
    while (true) {
        size_t end = input.find('\x1e', start);
        if (end == std::string::npos) end = input.size();
        for (const auto & piece : unicode_regex_split(input.substr(start, end - start), {regex}, false)) std::cout << piece << '\x1f';
        std::cout << '\x1e';
        if (end == input.size()) break;
        start = end + 1;
    }
}
