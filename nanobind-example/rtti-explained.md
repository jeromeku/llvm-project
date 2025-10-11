# RTTI (Run-Time Type Information) Explained

## Table of Contents

1. [What is RTTI?](#what-is-rtti)
2. [RTTI Features in C++](#rtti-features-in-c)
3. [Compiler Implementation](#compiler-implementation)
4. [Impact of -fno-rtti](#impact-of--fno-rtti)
5. [Binary Analysis](#binary-analysis)
6. [Workarounds Without RTTI](#workarounds-without-rtti)
7. [When to Disable RTTI](#when-to-disable-rtti)

---

## What is RTTI?

**RTTI (Run-Time Type Information)** is a C++ language feature that allows programs to determine the type of an object at runtime. It provides type safety for polymorphic types (classes with virtual functions).

### Core Mechanisms

RTTI provides three main facilities:

1. **`typeid` operator**: Get type information about an expression
2. **`dynamic_cast` operator**: Safely downcast pointers/references in inheritance hierarchies
3. **Exception handling**: Type-based exception catching

---

## RTTI Features in C++

### 1. The `typeid` Operator

Returns a reference to a `std::type_info` object that represents the type:

```cpp
Animal* animal = new Dog();

// Get type information
const std::type_info& ti = typeid(*animal);
std::cout << ti.name();        // Output: "3Dog" (mangled name)
std::cout << ti.hash_code();   // Unique hash for the type

// Compare types
if (typeid(*animal) == typeid(Dog)) {
    std::cout << "It's a Dog!";
}
```

**Key points:**
- `typeid` on a **pointer** gives you the pointer's static type
- `typeid` on a **dereferenced pointer** (`*ptr`) gives you the actual object's dynamic type
- Only works for polymorphic types (classes with virtual functions)

### 2. The `dynamic_cast` Operator

Safely downcasts pointers or references in an inheritance hierarchy:

```cpp
Animal* animal = get_some_animal();

// Safe downcast - returns nullptr if wrong type
if (Dog* dog = dynamic_cast<Dog*>(animal)) {
    dog->bark();  // Safe to use dog-specific methods
} else {
    // animal is not a Dog
}

// With references - throws std::bad_cast if wrong type
try {
    Dog& dog = dynamic_cast<Dog&>(*animal);
    dog.bark();
} catch (std::bad_cast& e) {
    // animal is not a Dog
}
```

**How it works:**
- Checks the actual runtime type via vtable
- Returns `nullptr` for pointers if cast fails
- Throws `std::bad_cast` for references if cast fails
- **Requires RTTI to be enabled**

### 3. Exception Handling with RTTI

RTTI enables type-based exception catching:

```cpp
try {
    throw DerivedException("error");
} catch (const DerivedException& e) {  // Catches DerivedException
    // Specific handler
} catch (const BaseException& e) {      // Would also catch DerivedException
    // General handler
}
```

The runtime uses RTTI to match exception types with catch blocks.

---

## Compiler Implementation

### What the Compiler Generates with RTTI Enabled

For each polymorphic class, the compiler generates:

#### 1. Type Info Structure (`typeinfo`)

```cpp
// Conceptual representation (actual structure is compiler-specific)
struct type_info {
    const char* name;           // Mangled type name
    const type_info* base_info; // Pointer to base class type_info
    // ... other metadata
};
```

#### 2. Enhanced VTable

The vtable contains a pointer to the type_info:

```
VTable for Dog:
+------------------------+
| offset_to_top          |
+------------------------+
| typeinfo for Dog   ←------- Points to type_info structure
+------------------------+
| Dog::~Dog()            |
+------------------------+
| Dog::speak()           |
+------------------------+
| Dog::get_name()        |
+------------------------+
```

#### 3. Type Info Names

Mangled names stored in read-only data:

```
typeinfo name for Dog: "3Dog"
typeinfo name for Cat: "3Cat"
typeinfo name for Animal: "6Animal"
```

The number prefix is the length of the class name.

---

## Impact of -fno-rtti

### What Gets Disabled

When compiling with `-fno-rtti`:

1. ❌ **`typeid` operator is disabled**
   ```cpp
   typeid(*animal)  // ERROR: cannot use 'typeid' with '-fno-rtti'
   ```

2. ❌ **`dynamic_cast` is disabled for polymorphic types**
   ```cpp
   dynamic_cast<Dog*>(animal)  // ERROR: 'dynamic_cast' not permitted with '-fno-rtti'
   ```

3. ⚠️ **Exception handling still works** (uses a different mechanism)
   ```cpp
   catch (const MyException& e)  // Still works!
   ```

4. ✅ **Virtual functions still work normally**
   ```cpp
   animal->speak()  // Still works via vtable
   ```

### What Gets Removed from Binary

Comparing our example binaries:

```bash
$ ls -lh rtti-example-*
-rwxrwxr-x 1 user user 36K  rtti-example-with-rtti
-rwxrwxr-x 1 user user 20K  rtti-example-no-rtti
```

**Size reduction: 44% smaller (16K saved)**

Detailed breakdown:

```bash
$ size rtti-example-with-rtti rtti-example-no-rtti
   text    data     bss     dec     hex filename
  15405    1360     280   17045    4295 rtti-example-with-rtti
   8770    1000     280   10050    2742 rtti-example-no-rtti
```

**What was removed:**
- **text segment**: -6,635 bytes (42% reduction) - Less code for RTTI operations
- **data segment**: -360 bytes (26% reduction) - No typeinfo structures

---

## Binary Analysis

### With RTTI Enabled

#### Symbols Present

```bash
$ nm -C rtti-example-with-rtti | grep -i "typeinfo\|vtable"
```

**typeinfo symbols:**
```
0000000000006c08 V typeinfo for AnimalException
0000000000006c38 V typeinfo for Cat
0000000000006c50 V typeinfo for Dog
0000000000006c20 V typeinfo for Bird
0000000000006c68 V typeinfo for Animal
```

**typeinfo name symbols:**
```
00000000000042d0 V typeinfo name for AnimalException
00000000000042e8 V typeinfo name for Cat
00000000000042ed V typeinfo name for Dog
00000000000042e2 V typeinfo name for Bird
00000000000042f8 V typeinfo name for Animal
```

**vtable symbols (include typeinfo pointers):**
```
0000000000006af0 V vtable for AnimalException
0000000000006b48 V vtable for Cat
0000000000006b78 V vtable for Dog
0000000000006b18 V vtable for Bird
0000000000006ba8 V vtable for Animal
```

#### Mangled Type Names

```bash
$ strings rtti-example-with-rtti | grep -E "^[0-9]+(Dog|Cat|Bird|Animal)"
15AnimalException    # Length prefix: 15 characters
3Cat                 # Length prefix: 3 characters
3Dog                 # Length prefix: 3 characters
4Bird                # Length prefix: 4 characters
6Animal              # Length prefix: 6 characters
```

### Without RTTI (-fno-rtti)

#### Symbols Present

```bash
$ nm -C rtti-example-no-rtti | grep -i "typeinfo\|vtable"
```

**No typeinfo symbols!** Only vtables:
```
0000000000003c80 V vtable for Cat
0000000000003cb8 V vtable for Dog
0000000000003c48 V vtable for Bird
0000000000003cf0 V vtable for Animal
```

**Key observation:** VTables are smaller - they don't contain typeinfo pointers.

### Memory Layout Comparison

#### With RTTI - VTable Structure

```
VTable for Dog:
+------------------------+
| offset_to_top          | -16 bytes
+------------------------+
| typeinfo for Dog*   ←------ RTTI pointer (8 bytes)
+------------------------+
| Dog::~Dog()            |  0 bytes
+------------------------+
| Dog::speak()           |  8 bytes
+------------------------+
| Dog::get_name()        | 16 bytes
+------------------------+
| Dog::get_type()        | 24 bytes
+------------------------+
```

#### Without RTTI - VTable Structure

```
VTable for Dog:
+------------------------+
| offset_to_top          | -8 bytes
+------------------------+
| Dog::~Dog()            |  0 bytes  (No typeinfo pointer!)
+------------------------+
| Dog::speak()           |  8 bytes
+------------------------+
| Dog::get_name()        | 16 bytes
+------------------------+
| Dog::get_type()        | 24 bytes
+------------------------+
```

**Result:** Each vtable is 8 bytes smaller (no typeinfo pointer).

---

## Workarounds Without RTTI

When RTTI is disabled, you need manual type identification:

### Pattern 1: Type Enum

```cpp
class Animal {
public:
    enum class Type { ANIMAL, DOG, CAT, BIRD };

    virtual ~Animal() = default;
    virtual Type get_type() const { return Type::ANIMAL; }
};

class Dog : public Animal {
public:
    Type get_type() const override { return Type::DOG; }
};

// Usage: Manual type checking
if (animal->get_type() == Animal::Type::DOG) {
    Dog* dog = static_cast<Dog*>(animal);  // Safe after check
    dog->bark();
}
```

**Pros:**
- Fast (single virtual function call)
- Small overhead (one vtable entry)
- Type-safe if used correctly

**Cons:**
- Must maintain enum manually
- Easy to forget to override `get_type()`
- Doesn't work across shared library boundaries well

### Pattern 2: Virtual Type Checking

```cpp
class Animal {
public:
    virtual ~Animal() = default;
    virtual bool is_dog() const { return false; }
    virtual bool is_cat() const { return false; }
};

class Dog : public Animal {
public:
    bool is_dog() const override { return true; }
};

// Usage
if (animal->is_dog()) {
    Dog* dog = static_cast<Dog*>(animal);
    dog->bark();
}
```

**Pros:**
- Explicit and clear
- Hard to forget (pure virtual in base)

**Cons:**
- Pollutes base class with type queries
- More vtable entries
- Not scalable for many types

### Pattern 3: Template-Based Type Checking

```cpp
template<typename T>
class TypedAnimal : public Animal {
public:
    static const void* type_id() {
        static char dummy;
        return &dummy;
    }

    const void* get_type_id() const override {
        return type_id();
    }
};

class Dog : public TypedAnimal<Dog> {
    // ...
};

// Usage
template<typename T>
T* animal_cast(Animal* animal) {
    if (animal->get_type_id() == TypedAnimal<T>::type_id()) {
        return static_cast<T*>(animal);
    }
    return nullptr;
}

if (Dog* dog = animal_cast<Dog>(animal)) {
    dog->bark();
}
```

**Pros:**
- Automatic type ID generation
- Type-safe
- Works across shared libraries (same binary)

**Cons:**
- More complex
- Template instantiation overhead
- Address uniqueness relies on linker behavior

### Pattern 4: Hash-Based Type ID

```cpp
class Animal {
public:
    virtual ~Animal() = default;
    virtual size_t type_hash() const { return typeid(*this).hash_code(); }  // With RTTI
    // virtual size_t type_hash() const = 0;  // Without RTTI - must implement
};

class Dog : public Animal {
    static constexpr size_t hash = compute_hash("Dog");
public:
    size_t type_hash() const override { return hash; }
};

// Compile-time hash function
constexpr size_t compute_hash(const char* str) {
    size_t hash = 5381;
    while (*str) {
        hash = ((hash << 5) + hash) + *str++;
    }
    return hash;
}
```

**Pros:**
- Compile-time hash computation
- Fast comparison
- Unique IDs

**Cons:**
- Hash collisions possible (though rare)
- Must maintain type names manually

---

## When to Disable RTTI

### Reasons to Disable RTTI (-fno-rtti)

1. **Binary Size Constraints**
   - Embedded systems with limited storage
   - Mobile apps where size matters
   - Typical savings: 5-15% of binary size

2. **Performance Considerations**
   - Slightly faster vtable lookups (no typeinfo pointer)
   - Reduced data cache pressure
   - Real-world impact: Usually negligible (< 1%)

3. **Code Style Enforcement**
   - Discourage `dynamic_cast` (can hide design issues)
   - Force explicit type handling
   - Encourage better OOP design

4. **ABI Compatibility**
   - Some libraries (like some STL implementations) may be compiled without RTTI
   - Must match RTTI settings across all linked code
   - Mixing RTTI/no-RTTI can cause undefined behavior

5. **Platform Requirements**
   - Some platforms (certain game consoles, embedded) forbid RTTI
   - Unreal Engine disables RTTI by default

### Reasons to Keep RTTI Enabled

1. **Using Standard Features**
   - Need `typeid` for type introspection
   - Need `dynamic_cast` for safe downcasting
   - Using libraries that require RTTI

2. **Debugging and Diagnostics**
   - Better exception messages with type names
   - Easier to debug polymorphic code
   - Runtime type inspection tools

3. **Serialization/Reflection**
   - Automatic object serialization
   - Reflection systems
   - Plugin systems that load unknown types

4. **Simplicity**
   - Less manual type management code
   - Standard C++ features work as expected
   - Easier for new team members

### nanobind and RTTI

**nanobind typically compiles with `-fno-rtti`** because:

1. **Size optimization**: Python bindings can be large; every byte counts
2. **Python's type system**: Python has its own runtime type system
3. **No need for dynamic_cast**: Type conversions are explicit in binding code
4. **ABI compatibility**: Consistent with common practices in binding libraries

If you need RTTI in your nanobind extension, you can enable it, but you must ensure:
- All linked code (including nanobind itself) is compiled with the same RTTI setting
- You understand the binary size impact

---

## Summary: RTTI Impact Comparison

| Aspect | With RTTI | Without RTTI (-fno-rtti) |
|--------|-----------|--------------------------|
| **Binary Size** | Larger (16K in our example) | Smaller (44% reduction) |
| **`typeid` operator** | ✅ Works | ❌ Compile error |
| **`dynamic_cast`** | ✅ Works | ❌ Compile error |
| **Exception handling** | ✅ Works | ✅ Works (different mechanism) |
| **Virtual functions** | ✅ Works | ✅ Works |
| **VTable size** | Larger (includes typeinfo*) | Smaller (no typeinfo*) |
| **Type comparison** | Automatic via `typeid` | Manual (enum/hash/etc.) |
| **Safe downcasting** | Automatic via `dynamic_cast` | Manual check + `static_cast` |
| **Cross-library types** | ✅ Works | ⚠️ Requires careful design |
| **Performance** | Slightly slower | Slightly faster |
| **Code complexity** | Simpler | More complex (manual type management) |
| **Type introspection** | Built-in | Must implement manually |

---

## Practical Example: The Difference

### Original Code (Requires RTTI)

```cpp
void process_animal(Animal* animal) {
    // Type identification
    std::cout << "Type: " << typeid(*animal).name() << "\n";

    // Safe downcasting
    if (Dog* dog = dynamic_cast<Dog*>(animal)) {
        dog->fetch();
    } else if (Cat* cat = dynamic_cast<Cat*>(animal)) {
        cat->scratch();
    }
}
```

### Refactored for No-RTTI

```cpp
class Animal {
public:
    enum class Type { DOG, CAT, BIRD };
    virtual ~Animal() = default;
    virtual Type get_type() const = 0;
    virtual const char* type_name() const = 0;
};

class Dog : public Animal {
public:
    Type get_type() const override { return Type::DOG; }
    const char* type_name() const override { return "Dog"; }
    void fetch();
};

void process_animal(Animal* animal) {
    // Manual type identification
    std::cout << "Type: " << animal->type_name() << "\n";

    // Manual downcasting
    if (animal->get_type() == Animal::Type::DOG) {
        Dog* dog = static_cast<Dog*>(animal);
        dog->fetch();
    } else if (animal->get_type() == Animal::Type::CAT) {
        Cat* cat = static_cast<Cat*>(animal);
        cat->scratch();
    }
}
```

### Binary Size Difference

For our example:
```bash
# With RTTI
-rwxrwxr-x 1 user user 36K  rtti-example-with-rtti

# Without RTTI
-rwxrwxr-x 1 user user 20K  rtti-example-no-rtti

# Savings: 16K (44% reduction)
```

In a real application with hundreds of polymorphic classes, the savings can be hundreds of kilobytes or even megabytes.

---

## Compilation Flags

### Enable RTTI (Default)
```bash
g++ -o myapp myapp.cpp                    # RTTI enabled by default
g++ -o myapp myapp.cpp -frtti            # Explicitly enable RTTI
```

### Disable RTTI
```bash
g++ -o myapp myapp.cpp -fno-rtti         # Disable RTTI
```

### Check if RTTI is Used
```bash
# List RTTI symbols
nm -C myapp | grep typeinfo

# Check binary size
size myapp

# Verify RTTI usage in source
grep -r "dynamic_cast\|typeid" src/
```

---

## Conclusion

RTTI is a powerful C++ feature that provides runtime type safety and introspection, but it comes with a cost in binary size and (minimal) performance. Understanding when to use it and when to disable it is crucial for:

- **Embedded systems**: Where binary size is critical
- **High-performance applications**: Where every byte of cache matters
- **Python bindings**: Where Python's type system replaces C++ RTTI
- **Large codebases**: Where consistent ABI compatibility is essential

When disabling RTTI, ensure you have alternative type identification mechanisms in place and that all linked code uses the same RTTI setting.
