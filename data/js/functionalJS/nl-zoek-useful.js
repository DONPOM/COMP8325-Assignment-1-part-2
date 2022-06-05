
        document.getElementById('pers_id').value='';
        var options = {
                script:"/zoek/auteurs.php?json=true&limit=15&",
                varname:"input",
                json:true,
                shownoresults:false,
                maxresults:15,
		timeout:100,
                callback: function (obj) { document.getElementById('pers_id').value = obj.id; }
        };
        var as_json_a = new bsn.AutoSuggest('auteur', options);
