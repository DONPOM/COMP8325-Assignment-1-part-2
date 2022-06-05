
    var makeURL = function(url){
        if(url.indexOf('//') == 0 && parent.location.protocol == "about:blank"){
            if(top.location.protocol == "http:") url = 'http:' + url;
            if(top.location.protocol == "https:") url = 'https:' + url;
        }
        else{
            if(url.indexOf('//') == 0 && parent.location.protocol == "http:") url = 'http:' + url;
            if(url.indexOf('//') == 0 && parent.location.protocol == "https:") url = 'https:' + url;
        }
        return url;

    }
    document.write('<a href="http://ad.rambler.ru/roff/ban.clk?pg=8983%26bn=240212%26cid=215616" target="_blank"><img src="' + makeURL('http://images.rambler.ru/upl/ad/2014/04/25/pixel.gif') + '" width="1" height="1" border="0"></a>');
