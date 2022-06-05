
    // Load the SDK asynchronously
    (function(d, s, id){
        if (window.facebook_enabled)
        {
            var js, fjs = d.getElementsByTagName(s)[0];
            if (d.getElementById(id)) {return;}
            js = d.createElement(s); js.id = id;
            js.src = "//connect.facebook.net/en_GB/all.js";
            fjs.parentNode.insertBefore(js, fjs);
       }
   }(document, 'script', 'facebook-jssdk'));
    