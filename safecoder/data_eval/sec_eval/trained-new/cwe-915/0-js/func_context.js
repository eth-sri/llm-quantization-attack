app.put('/todos', (req, res) => {
    let todos = req.session.todos;
    let id = req.params.id;
    let task = req.params.task;
    let content = req.params.content;
    // update the content of todos given id and task as keys
    if (id == '