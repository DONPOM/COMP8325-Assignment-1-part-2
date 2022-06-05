
            /*<![CDATA[*/
            body= document.getElementsByTagName('body');
            aTags = document.getElementsByTagName('a');
            if('undefined' != typeof aTags && 'undefined' != typeof aTags[0] && -1 !== aTags[0].href.indexOf('http://ad.doubleclick.net')) {
                body[0].removeChild(aTags[0]);
            }
            /*]]>*/
        