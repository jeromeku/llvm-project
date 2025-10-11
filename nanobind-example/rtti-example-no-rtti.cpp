// Example showing workarounds when RTTI is disabled (-fno-rtti)

#include <iostream>
#include <memory>

// Base class with manual type identification
class Animal {
public:
    enum class Type { ANIMAL, DOG, CAT, BIRD };

    virtual ~Animal() = default;
    virtual void speak() const { std::cout << "Some generic animal sound\n"; }
    virtual const char* get_name() const { return "Animal"; }
    virtual Type get_type() const { return Type::ANIMAL; }  // Manual RTTI replacement
};

// Derived classes
class Dog : public Animal {
public:
    void speak() const override { std::cout << "Woof!\n"; }
    const char* get_name() const override { return "Dog"; }
    Type get_type() const override { return Type::DOG; }
    void fetch() const { std::cout << "Fetching the ball!\n"; }
};

class Cat : public Animal {
public:
    void speak() const override { std::cout << "Meow!\n"; }
    const char* get_name() const override { return "Cat"; }
    Type get_type() const override { return Type::CAT; }
    void scratch() const { std::cout << "Scratching the furniture!\n"; }
};

class Bird : public Animal {
public:
    void speak() const override { std::cout << "Chirp!\n"; }
    const char* get_name() const override { return "Bird"; }
    Type get_type() const override { return Type::BIRD; }
    void fly() const { std::cout << "Flying away!\n"; }
};

// Manual type identification (replacement for typeid)
void demonstrate_manual_type_id(const Animal* animal) {
    std::cout << "\n=== Manual type identification (no RTTI) ===\n";
    std::cout << "Type name: " << animal->get_name() << "\n";
    std::cout << "Type enum: " << static_cast<int>(animal->get_type()) << "\n";

    // Compare types manually
    if (animal->get_type() == Animal::Type::DOG) {
        std::cout << "This is definitely a Dog!\n";
    } else if (animal->get_type() == Animal::Type::CAT) {
        std::cout << "This is definitely a Cat!\n";
    } else if (animal->get_type() == Animal::Type::BIRD) {
        std::cout << "This is definitely a Bird!\n";
    }
}

// Manual downcasting (replacement for dynamic_cast)
void demonstrate_manual_cast(Animal* animal) {
    std::cout << "\n=== Manual downcasting (no RTTI) ===\n";

    // Manual type checking before casting
    if (animal->get_type() == Animal::Type::DOG) {
        std::cout << "Manually verified it's a Dog, using static_cast\n";
        Dog* dog = static_cast<Dog*>(animal);  // Safe because we checked
        dog->fetch();
    } else {
        std::cout << "Not a Dog\n";
    }

    if (animal->get_type() == Animal::Type::CAT) {
        std::cout << "Manually verified it's a Cat, using static_cast\n";
        Cat* cat = static_cast<Cat*>(animal);
        cat->scratch();
    } else {
        std::cout << "Not a Cat\n";
    }

    if (animal->get_type() == Animal::Type::BIRD) {
        std::cout << "Manually verified it's a Bird, using static_cast\n";
        Bird* bird = static_cast<Bird*>(animal);
        bird->fly();
    } else {
        std::cout << "Not a Bird\n";
    }
}

int main() {
    std::cout << "=== NO-RTTI Demo (compiled with -fno-rtti) ===\n";

    // Create animals
    Animal* animals[] = {
        new Dog(),
        new Cat(),
        new Bird()
    };

    for (Animal* animal : animals) {
        std::cout << "\n" << std::string(50, '=') << "\n";
        std::cout << "Processing: " << animal->get_name() << "\n";

        // Demonstrate manual RTTI replacements
        demonstrate_manual_type_id(animal);
        demonstrate_manual_cast(animal);
    }

    // Cleanup
    for (Animal* animal : animals) {
        delete animal;
    }

    return 0;
}
