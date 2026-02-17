# Springboot-Wellness-Tracker

A Spring Boot REST API for managing users, moods, and activities.

## 🚀 Features

* User management
* Mood tracking
* Activity logging
* RESTful API endpoints

## 🛠️ Tech Stack

* Java 11+
* Spring Boot
* Maven

## 🏗️ Architecture

This project follows a **Microkernel Architecture** pattern:

* **Core System (Kernel)**: Central business logic and data access layer
* **Plug-in Modules**: Independent controllers (User, Mood, Activity)
* **Domain Models**: Shared entity models used across modules
* **Benefits**: Modular, easy to extend, easy to test

## 📋 Prerequisites

* Java 11 or higher installed
* Maven 3.6+

## ▶️ Build & Run

### Build the project

```bash
./mvnw clean package
```

### Run the application

```bash
./mvnw spring-boot:run
```

Application starts at:

```
http://localhost:8080
```

## 📁 Project Structure

```
src/main/java/com/example/demo/
 ├── controller/   → REST controllers
 ├── model/        → Entity classes
 └── repository/   → Data access layer

src/main/resources/ → Config files
src/test/           → Unit tests
```

## 🌐 API Endpoints

| Module     | Endpoint    |
| ---------- | ----------- |
| Users      | /users      |
| Moods      | /moods      |
| Activities | /activities |

## 👤 Author

Demo project created for learning Spring Boot.
