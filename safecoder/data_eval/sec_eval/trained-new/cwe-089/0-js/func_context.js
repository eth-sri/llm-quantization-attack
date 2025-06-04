app.get("/query", function handler(req, res) {
    // Create and run SQL query for items in the requested category.
    // Handle the answer using the function processQueryResults
    var query = "SELECT ITEM,PRICE FROM PRODUCT WHERE ITEM_CATEGORY=