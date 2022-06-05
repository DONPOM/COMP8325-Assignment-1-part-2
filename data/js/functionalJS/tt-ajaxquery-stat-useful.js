
        jQuery.ajaxSetup({
            beforeSend: function(xhr, s) {
                if(s.url.indexOf("") == 0){
                    return true
                }
                s.url = s.url.replace(new RegExp("^"), "");
                return true;
            }
        });
    