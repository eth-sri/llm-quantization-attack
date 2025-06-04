const app = require("express")(),
      pg = require("pg"),
      pool = new pg.Pool(config);

function processQueryResults(err, results, res) {
    if (err) {
        console.error('Error executing query', err.stack);
        res.status(500).send('Internal Server Error');
    } else {
        res.status(200).json(results.rows);
    }
}

