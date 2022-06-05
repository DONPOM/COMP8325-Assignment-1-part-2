
    connect(window, 'onDOMload', function(e){
        window.undo_delete = function(project_id) {
           var req = new xJSONscriptRequest('/p' + project_id + '/delete_project_json/?undo=true');
           req.buildScriptTag();
           req.addScriptTag();
        }
        window.delete_project = function(project_id){
           var req = new xJSONscriptRequest('/p' + project_id + '/delete_project_json/');
           req.buildScriptTag();
           req.addScriptTag();
        }
    });
	