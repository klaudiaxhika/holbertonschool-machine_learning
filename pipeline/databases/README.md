Let's break down your questions one by one:

### 1. **Relational Database:**
A relational database is a type of database that uses a structure based on a formal system of relationships. It organizes data into tables, where each table consists of rows and columns. These tables are related to each other based on predefined relationships to ensure data integrity and consistency.

### 2. **Non-Relational Database (NoSQL):**
Non-relational databases, or NoSQL databases, are databases that do not use the traditional SQL-based relational model. They are designed to handle large volumes of unstructured or semi-structured data. NoSQL databases use a variety of data models, including document, key-value, column-family, and graph.

### 3. **Difference Between SQL and NoSQL:**
SQL databases are relational databases that use structured query language (SQL) for defining and manipulating the data. NoSQL databases, on the other hand, do not require a fixed schema and are best suited for large sets of distributed data.

### 4. **Creating Tables with Constraints (in SQL):**
To create tables with constraints in SQL, you can use the CREATE TABLE statement with various constraints like PRIMARY KEY, FOREIGN KEY, UNIQUE, NOT NULL, etc.

Example:
```sql
CREATE TABLE Users (
    UserID INT PRIMARY KEY,
    UserName VARCHAR(255) NOT NULL,
    Email VARCHAR(255) UNIQUE
);
```

### 5. **Optimizing Queries by Adding Indexes (in SQL):**
Indexes in databases help optimize query performance. You can add an index to a column using the CREATE INDEX statement.

Example:
```sql
CREATE INDEX idx_user_id ON Users(UserID);
```

### 6. **Stored Procedures and Functions in MySQL:**
Stored procedures and functions are precompiled collections of one or more SQL statements. Stored procedures are used for performing operations, and functions return a value.

Example of creating a stored procedure:
```sql
DELIMITER //
CREATE PROCEDURE GetUserInfo(IN userID INT)
BEGIN
    SELECT * FROM Users WHERE UserID = userID;
END //
DELIMITER ;
```

### 7. **Views in MySQL:**
A view is a virtual table based on the result of a SELECT query. It provides a way to simplify complex queries by encapsulating them into a view.

Example of creating a view:
```sql
CREATE VIEW UserEmails AS
SELECT UserID, Email FROM Users;
```

### 8. **Triggers in MySQL:**
A trigger is a set of instructions that are automatically executed ("triggered") in response to certain events on a particular table in a database.

Example of creating a trigger:
```sql
CREATE TRIGGER after_insert_user
AFTER INSERT ON Users
FOR EACH ROW
BEGIN
    -- Trigger logic here
END;
```

### 9. **ACID (Atomicity, Consistency, Isolation, Durability):**
ACID is a set of properties that guarantee that database transactions are processed reliably. ACID stands for Atomicity (transactions are all or nothing), Consistency (transactions bring the database from one valid state to another), Isolation (transactions are isolated from each other until they are complete), and Durability (once a transaction is committed, it is permanent).

### 10. **Document Storage:**
Document storage in NoSQL databases refers to storing data in a format like JSON or BSON (binary JSON). Each document is a set of key-value pairs, and documents can be nested within each other.

### 11. **NoSQL Types:**
There are several types of NoSQL databases:
- **Document Stores:** Store data in document format (e.g., MongoDB).
- **Key-Value Stores:** Use keys to access values (e.g., Redis).
- **Column-Family Stores:** Store data in columns rather than rows (e.g., Apache Cassandra).
- **Graph Databases:** Designed for handling complex relationships (e.g., Neo4j).

### 12. **Benefits of NoSQL Databases:**
- **Scalability:** NoSQL databases can handle large amounts of data and traffic.
- **Flexibility:** They can handle unstructured or semi-structured data.
- **High Performance:** Especially for read and write operations on large datasets.
- **Availability and Fault Tolerance:** NoSQL databases are often designed to be distributed and fault-tolerant.

### 13. **Querying, Inserting, Updating, Deleting in NoSQL (MongoDB):**
In MongoDB, you use methods like `find()`, `insertOne()`, `updateOne()`, `deleteOne()` for querying, inserting, updating, and deleting documents, respectively.

Example of querying:
```javascript
db.collection.find({ key: "value" });
```

Example of inserting:
```javascript
db.collection.insertOne({ key: "value" });
```

Example of updating:
```javascript
db.collection.updateOne({ key: "value" }, { $set: { newKey: "newValue" } });
```

Example of deleting:
```javascript
db.collection.deleteOne({ key: "value" });
```

### 14. **Using MongoDB:**
To use MongoDB, you need to install MongoDB on your system. Then you can start the MongoDB server and interact with it through the MongoDB shell or a programming language like JavaScript (Node.js).

Example of connecting to MongoDB using Node.js:
```javascript
const mongoose = require('mongoose');

mongoose.connect('mongodb://localhost:27017/mydatabase', {useNewUrlParser: true, useUnifiedTopology: true});

const db = mongoose.connection;
db.on('error', console.error.bind(console, 'connection error:'));
db.once('open', function() {
  console.log('Connected to MongoDB');
});
```

Make sure to replace `'mongodb://localhost:27017/mydatabase'` with the appropriate connection URL for your MongoDB instance.

Remember that MongoDB syntax and usage might evolve, so it's always a good idea to refer to the official MongoDB documentation for the most up-to-date information.
