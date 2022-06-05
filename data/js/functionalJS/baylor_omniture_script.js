
 /* You may give each page an identifying name, server, and channel on
    the next lines. */
    s.pageName = ""
    s.server = ""
    s.channel = ""
    s.pageType = ""
    s.prop1 = ""
    s.prop2 = ""
    s.prop3 = ""
    s.prop4 = ""
    s.prop5 = ""
    s.prop6 = ""
    /* Conversion Variables */
    s.campaign = ""
    s.state = ""
    s.zip = ""
    s.events = ""
    s.products = ""
    s.purchaseID = ""
    s.eVar1 = ""
    s.eVar2 = ""
    s.eVar3 = ""
    s.eVar4 = ""
    s.eVar5 = ""

    s.prop7 = "";
    s.prop8 = "";
    s.prop9 = "";
    s.prop14 = "";
    s.prop15 = "";

    /**BAYLOR CUSTOM CODE**/
    //assigns page and section vars
    $(function() {



        $('.ms-sitemapdirectional').each(function(intIndex) {
            //intIndex = 0 will always be "Baylor Health Care System". Don't count this level.
            if (intIndex > 0) {
                //imitate the property name for this iteration
                var propName = 's.prop' + (intIndex + 2);

                //if we are deeper in the site heirarchy than our variables allow, don't count the level
                if (typeof (eval(propName)) != 'undefined') {
                    //set the var
                    eval('s.prop' + (intIndex + 2) + '=\'' + $(this).html().replace("'", "") + '\'');
                }
            }

        })

        //if each property has length > 0, append it to the s.pagename var.
        if (s.prop3.length > 0) { s.pageName += s.prop3; }
        if (s.prop4.length > 0) { s.pageName += '>' + s.prop4; }
        if (s.prop5.length > 0) { s.pageName += '>' + s.prop5; }
        if (s.prop6.length > 0) { s.pageName += '>' + s.prop6; }
        if (s.prop7.length > 0) { s.pageName += '>' + s.prop7; }
        if (s.prop8.length > 0) { s.pageName += '>' + s.prop8; }
        if (s.prop9.length > 0) { s.pageName += '>' + s.prop9; }
        if (s.prop14.length > 0) { s.pageName += '>' + s.prop14; }
        if (s.prop15.length > 0) { s.pageName += '>' + s.prop15; }
    

        var s_code = s.t(); if (s_code) document.write(s_code);


    });
    if (navigator.appVersion.indexOf('MSIE') >= 0) document.write(unescape('%3C') + '\!-' + '-');



    /**END BAYLOR CUSTOM CODE**/


