import gov.va.vinci.leo.cr.BatchDatabaseCollectionReader;

reader = new BatchDatabaseCollectionReader("com.mysql.jdbc.Driver", "jdbc:mysql://localhost/test",
        "testuser", "password", "select id, document from example_document", "id", "document", 0,
        100, 10);

