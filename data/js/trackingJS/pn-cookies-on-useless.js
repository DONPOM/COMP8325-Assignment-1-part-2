
    var fb_validated = false;
    window.fbAsyncInit = function () {
        FB.init({
            appId: '130127590512253', // App ID
            channelURL: '//www.patreon.com/channel.php', // Channel File
            status: false, // check login status
            cookie: true, // enable cookies to allow the server to access the session
            oauth: true, // enable OAuth 2.0
            xfbml: true,  // parse XFBML
            frictionlessRequests: true
        });

        FB.getLoginStatus(function (response) {
            if (response.status === 'connected') {
                /*FB.api({ method: 'fql.query',query: 'SELECT publish_stream FROM permissions WHERE uid=me()' },
                 function(response) {
                 if(response[0].publish_stream == 1)
                 fb_validated = true;
                 }
                 );*/
                fb_validated = true;
                            }
        });
        // Additional initialization code here
    };

    // Load the SDK Asynchronously
    (function (d, s, id) {
        var js, fjs = d.getElementsByTagName(s)[0];
        if (d.getElementById(id)) {
            return;
        }
        js = d.createElement(s);
        js.id = id;
        js.src = "//connect.facebook.net/en_US/all.js";
        fjs.parentNode.insertBefore(js, fjs);
    }(document, 'script', 'facebook-jssdk'));
