
    setTimeout(function(){
        try{
            //sis3.0 pixel
            var url_sis3 = 'http://s.amazon-adsystem.com/iu3?',
                params_sis3 = [
                    "d=imdb.com",
                    "a1=",
                    "a2=0101210b502a73b42b2aa4c57ef25c648bd7578df4bd06784bc66049d68ce4c17e69",
                    "pId=",
                    "r=1",
                    "rP=http%3A%2F%2Fwww.imdb.com%2F",
                    "encoding=server",
                    "cb=717322378532"  
                ];
        
            (document.getElementById('sis_pixel_sitewide')).src = url_sis3 + params_sis3.join('&');
        }
        catch(e){
            if ('consoleLog' in window){
                consoleLog('Pixel failure ' + e.toString(),'sis');
            }
            if (window.ueLogError) { 
                window.ueLogError(e);
            }
        }
    }, 5);
