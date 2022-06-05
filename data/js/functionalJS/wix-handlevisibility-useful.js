
        function handleVisibilityChange(evt)
        {
            var frog_img = new Image(); frog_img.src = "http://frog.wix.com/pblc?evid=2";
        }
        var browser_info = (function ()
        {
            var ua = navigator.userAgent,
            N = navigator.appName,tem,
            M = ua.match(/(opera|chrome|safari|firefox|msie|trident)\/?\s*([\d\.]+)/i) || [];
            M = M[2] ? [M[1],M[2]] : [N,navigator.appVersion,'-?'];
            if (M && (tem = ua.match(/version\/([\.\d]+)/i)) != null) M[2] = tem[1];
            return M.join(' ');
        })();
        function ResolutionManager()
        {
            this.steps = [1280,1600,1920,2560];
            this.min_steps = [0,1280,1600,1920];
            this.screenWidth = window.screen.availWidth;
            this.width = document.getElementsByTagName('html')[0].clientWidth;
            this.currentStepInx = 2;
            for (var i = 0; i < this.steps.length; i++)
            {
                if (this.width <= this.steps[i])
                {
                    this.currentStepInx = i;
                    break;
                }
            }
        }
        var resolutionManager = new ResolutionManager();
        var slot_rand = Date.now().toString();
        slot_rand = parseInt(slot_rand[slot_rand.length - 1]);
        var photo_slot = slot_rand < 3 ? ["_6","_13","_15","_3","_0","_14"] : slot_rand < 6 ? ["_4","_1","_2","_9","_10","_16"] : ["_14","_8","_4","_12","_11","_7"];
        var low_res_image = photo_slot[0].substring(1) + ".jpg";
        var dmy = new Image(10,10);
        if (resolutionManager.currentStepInx > 2)
        {
            dmy.src = "http://static.parastorage.com/services/html-landing/hp/brazil/images/low_res_2_2560/" + low_res_image;
        } else
        {
            dmy.src = "http://static.parastorage.com/services/html-landing/hp/brazil/images/low_res/" + low_res_image;
        }
        var pepoleDir = "wix_users2/";
        var staticHostImages = "http://static.parastorage.com/services/html-landing/hp/brazil/images/";
        var numOfImages = photo_slot.length;
        window.___gcfg = { lang: 'en-GB' };
        if (window.msWriteProfilerMark)
        {
            document.getElementsByTagName('html')[0].setAttribute('iefontsloaded','false');
        }

        var low_res_css = '<style type="text/css" > \
        @media (min-width: 0px) and (max-width:1280px){ .stage_1 .bk[inx=_dbl]{background-image:url(http://static.parastorage.com/services/html-landing/hp/brazil/images/1280/stage_1/wix_users2/' + low_res_image + '),url(http://static.parastorage.com/services/html-landing/hp/brazil/images/low_res/' + low_res_image + ');}} \
        @media (min-width: 1281px) and (max-width:1600px){ .stage_1 .bk[inx=_dbl]{background-image:url(http://static.parastorage.com/services/html-landing/hp/brazil/images/1600/stage_1/wix_users2/' + low_res_image + '),url(http://static.parastorage.com/services/html-landing/hp/brazil/images/low_res/' + low_res_image + ');}} \
        @media (min-width: 1601px) and (max-width:1920px){ .stage_1 .bk[inx=_dbl]{background-image:url(http://static.parastorage.com/services/html-landing/hp/brazil/images/1920/stage_1/wix_users2/' + low_res_image + '),url(http://static.parastorage.com/services/html-landing/hp/brazil/images/low_res/' + low_res_image + ');}} \
        @media (min-width: 1921px) and (max-width:6500px){ .stage_1 .bk[inx=_dbl]{background-image:url(http://static.parastorage.com/services/html-landing/hp/brazil/images/2560/stage_1/wix_users2/' + low_res_image + '),url(http://static.parastorage.com/services/html-landing/hp/brazil/images/low_res_2_2560/' + low_res_image + ');}} \
        </style>';
        document.writeln(low_res_css);
        if (browser_info.indexOf("MSIE") >= 0)
        {
            is_ie = true;
        }
        var submenu = null;
        var my_account_url;
        var postreg_url;
        var lng = 'en';
        var sub_lng = 'www';
        var selectedLanguage = "en";
        var wix_on_fb = "http%3A%2F%2Fwww.facebook.com/wix";
        var userApi;
        var wixTwitter = "I'm checking out http://www.Wix.com, the free website builder via @wix #MyNextWebsite";
        var video_data = { url: "http://static.wix.com/services/html-landing/hp/brazil/videos/en/video.flv",
            thumb: "http://static.wix.com/services/html-landing/hp/brazil/videos/en/thumb.jpg",
            player: "http://static.wix.com/services/html-landing/2013/videos/wixvideo_purple.swf?v=6",
            width: "800",
            height: "450",
            title: "Wix Tour",
            auto_play: "true",
            youtube_video: "http://www.youtube.com/embed/RintSKngKL4"
        };
        function FontLoadDetect(fonts,callback)
        {
            var interval = null;
            var w = [];
            var s = [];
            function checkFont()
            {
                for (var i = 0; i < w.length; i++)
                {
                    if (s[i].offsetWidth == w[i])
                    {
                        return;
                    }
                }
                clearInterval(interval);
                callback();
            };
            for (var i = 0; i < fonts.length; i++)
            {
                s.push(document.createElement('span'));
                s[i].className = 'dummy_spans';
                s[i].innerHTML = 'giItT1WQy@!-/#';
                s[i].style.fontFamily = 'non-existing-font';
                document.body.appendChild(s[i]);
                w.push(s[i].offsetWidth);
                s[i].style.fontFamily = fonts[i];
            }
            interval = setInterval(checkFont,30);
        }
    