
        document.getElementById('ti_id').value='';
        var options = {
                script:"/zoek/titels.php?json=true&limit=15&",
                varname:"input",
                json:true,
                shownoresults:false,
                maxresults:15,
                callback: function (obj) { document.getElementById('ti_id').value = obj.id; }
        };
        var as_json_t = new bsn.AutoSuggest('titel', options);
