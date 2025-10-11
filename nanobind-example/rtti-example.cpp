// Example demonstrating RTTI (Run-Time Type Information)

#include <iostream>
#include <typeinfo>
#include <memory>

// Base class
class Animal {
public:
    virtual ~Animal() = default;
    virtual void speak() const { std::cout << "Some generic animal sound\n"; }
    virtual const char* get_name() const { return "Animal"; }
};

// Derived classes
class Dog : public Animal {
public:
    void speak() const override { std::cout << "Woof!\n"; }
    const char* get_name() const override { return "Dog"; }
    void fetch() const { std::cout << "Fetching the ball!\n"; }
};

class Cat : public Animal {
public:
    void speak() const override { std::cout << "Meow!\n"; }
    const char* get_name() const override { return "Cat"; }
    void scratch() const { std::cout << "Scratching the furniture!\n"; }
};

class Bird : public Animal {
public:
    void speak() const override { std::cout << "Chirp!\n"; }
    const char* get_name() const override { return "Bird"; }
    void fly() const { std::cout << "Flying away!\n"; }
};

// Functions demonstrating RTTI usage

// 1. typeid operator - Get type information
void demonstrate_typeid(const Animal* animal) {
    std::cout << "\n=== typeid() demonstration ===\n";
    std::cout << "Type name: " << typeid(*animal).name() << "\n";
    std::cout << "Hash code: " << typeid(*animal).hash_code() << "\n";

    // Compare types
    if (typeid(*animal) == typeid(Dog)) {
        std::cout << "This is definitely a Dog!\n";
    } else if (typeid(*animal) == typeid(Cat)) {
        std::cout << "This is definitely a Cat!\n";
    } else if (typeid(*animal) == typeid(Bird)) {
        std::cout << "This is definitely a Bird!\n";
    }
}

// 2. dynamic_cast - Safe downcasting
void demonstrate_dynamic_cast(Animal* animal) {
    std::cout << "\n=== dynamic_cast<> demonstration ===\n";

    // Try to cast to Dog
    if (Dog* dog = dynamic_cast<Dog*>(animal)) {
        std::cout << "Successfully cast to Dog!\n";
        dog->fetch();
    } else {
        std::cout << "Not a Dog\n";
    }

    // Try to cast to Cat
    if (Cat* cat = dynamic_cast<Cat*>(animal)) {
        std::cout << "Successfully cast to Cat!\n";
        cat->scratch();
    } else {
        std::cout << "Not a Cat\n";
    }

    // Try to cast to Bird
    if (Bird* bird = dynamic_cast<Bird*>(animal)) {
        std::cout << "Successfully cast to Bird!\n";
        bird->fly();
    } else {
        std::cout << "Not a Bird\n";
    }
}

// 3. Practical use case: Factory pattern with type checking
Animal* create_random_animal(int choice) {
    switch (choice % 3) {
        case 0: return new Dog();
        case 1: return new Cat();
        case 2: return new Bird();
        default: return new Animal();
    }
}

// 4. Exception handling with RTTI
class AnimalException : public std::exception {
    std::string message;
public:
    explicit AnimalException(const std::string& msg) : message(msg) {}
    const char* what() const noexcept override { return message.c_str(); }
};

void demonstrate_exception_catch() {
    std::cout << "\n=== Exception handling with RTTI ===\n";
    try {
        throw AnimalException("Something went wrong with the animal!");
    } catch (const AnimalException& e) {
        // RTTI allows catching specific exception types
        std::cout << "Caught AnimalException: " << e.what() << "\n";
        std::cout << "Exception type: " << typeid(e).name() << "\n";
    } catch (const std::exception& e) {
        std::cout << "Caught generic exception\n";
    }
}

int main() {
    std::cout << "=== RTTI (Run-Time Type Information) Demo ===\n";

    // Create animals
    Animal* animals[] = {
        new Dog(),
        new Cat(),
        new Bird()
    };

    for (Animal* animal : animals) {
        std::cout << "\n" << std::string(50, '=') << "\n";
        std::cout << "Processing: " << animal->get_name() << "\n";

        // Demonstrate RTTI features
        demonstrate_typeid(animal);
        demonstrate_dynamic_cast(animal);
    }

    // Exception handling
    demonstrate_exception_catch();

    // Cleanup
    for (Animal* animal : animals) {
        delete animal;
    }

    return 0;
}
