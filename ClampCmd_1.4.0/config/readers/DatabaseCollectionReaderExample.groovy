import gov.va.vinci.leo.cr.DatabaseCollectionReader;

reader = new DatabaseCollectionReader("com.mysql.jdbc.Driver", "jdbc:mysql://localhost/test",
        "testuser", "password", "select id, document from example_document", "id", "document");