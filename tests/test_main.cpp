#include <iostream>

#include "test_framework.h"
#include "todo.h"

int main() {
    int passed = 0, failed = 0, todo = 0;
    for (const auto& test : wbtest::registry()) {
        try {
            test.body();
            ++passed;
            std::cout << "PASS  " << test.name << "\n";
        } catch (const wb::NotImplemented& stub) {
            ++todo;
            std::cout << "TODO  " << test.name << "  (needs " << stub.what() << ")\n";
        } catch (const std::exception& error) {
            ++failed;
            std::cout << "FAIL  " << test.name << "\n      " << error.what() << "\n";
        }
    }
    std::cout << "\n" << passed << " passed, " << failed << " failed, " << todo << " todo\n";
    return failed == 0 ? 0 : 1;
}
