// Site Search - Master Header 
    function SearchClick()
    {
        var searchUrl = 'http://search.baylorhealth.com/search?q=' + document.getElementById('idSearchString').value + '&btnG=Search&entqr=0&ud=1&sort=&output=xml_no_dtd&oe=UTF-8&ie=UTF-8&client=newconsumer_frontend&proxystylesheet=newconsumer_frontend&site=default_collection';
        window.location.href= searchUrl;
    }
    
    function TrapEnterKey(event)
    {
        var searchUrl = 'http://search.baylorhealth.com/search?q=' + document.getElementById('idSearchString').value + '&btnG=Search&entqr=0&ud=1&sort=&output=xml_no_dtd&oe=UTF-8&ie=UTF-8&client=newconsumer_frontend&proxystylesheet=newconsumer_frontend&site=default_collection';
        if (document.all)
        {
            if (event.keyCode == 13)
            {
                event.returnValue=false;
                event.cancel = true;
                window.location.href= searchUrl;
            }
        }
        else if (document.getElementById)
        {
            if (event.which == 13)
            {
                event.preventDefault();
                window.location.href= searchUrl;
            }
        }
        else if(document.layers)
        {
            if(event.which == 13)
            {
                event.returnValue=false;
                event.cancel = true;
                window.location.href= searchUrl;
            }
        }
    }

