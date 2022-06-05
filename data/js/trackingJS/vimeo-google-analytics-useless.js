
            // Analytics Setup
        var _comscore = _comscore || [],
            _gaq = _gaq || [];

        _comscore.push({
            c1: "2",
            c2: "10348289"
        });

        _gaq.push(['_require', 'inpage_linkid', '//www.google-analytics.com/plugins/ga/inpage_linkid.js']);
        _gaq.push(['_setAccount', "UA-76641-8"]);
        _gaq.push(['_gat._anonymizeIp']);

                    _gaq.push(['_setDomainName', 'vimeo.com']);
        
        _gaq.push(['_setAllowLinker', true]);
        _gaq.push(['_setAllowAnchor', true]);

        
        function trackHouseAd() {
            var ga_l = document.location;
            var ga_r = document.referrer;
            var ga_s = ga_l.search;

            if (ga_s.indexOf('house_ad') > -1) {
                var house_ad_value = ga_s.match(/house_ad=([^&]+)/)[1];

                _gaq.push(['_trackEvent', 'engagement:house-ad', house_ad_value, ga_r, undefined, true]);
                _gaq.push(['_setCustomVar', 8, 'house_ad', house_ad_value, 1]);
            }
        }

        trackHouseAd();

                    _gaq.push(['_setCustomVar', 2, 'user_type', 'basic', 1]);
            _gaq.push(['_setCustomVar', 7, 'video_count', '0', 1]);
        
        _gaq.push(['_setCustomVar', 1, 'user_status', 'logged_in', 3]);
        _gaq.push(['_setCustomVar', 4, 'language', 'en', 3]);

        
        _gaq.push(['_trackPageview']);

        
            
    var CSS_DIR = 'https://f.vimeocdn.com/styles/css_opt/',
        JS_DIR = 'https://f.vimeocdn.com/js_opt/',
        IMAGE_DIR = 'https://f.vimeocdn.com/images_v6/',
                BUILD_HASH = '5deeb',
        vimeo = {
            app_version: 'v6',
            domain: '.vimeo.com',
            url: 'vimeo.com',
            xsrft: '1553da4996985525d0e1224c3d338993.310550258d8614d276c7ae645f5634d6',
            cur_user: null,
            config: {},
            party_pooped: null        };

            vimeo.cur_user = {"id":34598903,"url":"\/user34598903","display_name":"Ikram Khan","image_url":{"30":"https:\/\/secure.gravatar.com\/avatar\/2db53a769303ac3d94dd49675fa32d6f?d=https%3A%2F%2Fi.vimeocdn.com%2Fportrait%2Fdefault-red_30x30.png&s=30"},"account_type":"","opted_out":false,"is_verified":true}    
    
    var localeConfig = {
        'Date': {
            months: ["January","February","March","April","May","June","July","August","September","October","November","December"],
            months_abbr: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            days: ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"],
            days_abbr: ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"],

            // Culture's date order: MM/DD/YYYY
            dateOrder: ['date', 'month', 'year'],
            shortDate: '%d/%m/%Y',
            shortTime: '%I:%M%p',
            AM: 'AM',
            PM: 'PM',
            firstDayOfWeek: 0,

            // Date.Extras
            ordinal: function(dayOfMonth){
                return dayOfMonth;
            }
        },
        'DatePicker': {
            select_a_time: "Select a time",
            use_mouse_wheel: "Use the mouse wheel to quickly change value",
            time_confirm_button: "OK",
            apply_range: "Apply",
            cancel: "Cancel",
            week: "Wk"        },
                'Number': {
            decimal: '.',
            group: ',',
            currency: {
                prefix: '$'
            }
        },
        'FormValidator': {"required":"This field is required.","requiredChk":"This field is required."}    };

    var fullLocale = '';

    
    var Copy = {
        translate: function(key, plural, replacements) {
            var translation = typeof this.dict[key] != 'object' ? this.dict[key] : (plural ? this.dict[key].plural : this.dict[key].singular);
            if (typeof replacements === 'object') {
                translation = this.substitute(translation, replacements);
            }
            return translation;
        },
        substitute: function(string, object) {
            if (!string) {
                return;
            }
            if (typeof string.substitute !== 'undefined') {
                return string.substitute(object);
            }
            else {
                return string.replace(/\\?\{([^{}]+)\}/g, function(match, name) {
                    if (match.charAt(0) == '\\') return match.slice(1);
                    return (object[name] != null) ? object[name] : '';
                });
            }
        },
        dict: {"did_you_mean_email":"Did you mean <em>{SUGGEST}<\/em>?","email_password_mismatch":"Email and password do not match","just_now":"just now","seconds_ago":{"singular":"{COUNT} second ago","plural":"{COUNT} seconds ago"},"minutes_ago":{"singular":"{COUNT} minute ago","plural":"{COUNT} minutes ago"},"hours_ago":{"singular":"{COUNT} hour ago","plural":"{COUNT} hours ago"},"open_comment_box":"Add new comment instead &raquo;","url_unavailable":"Sorry, this url is unavailable.","unsaved_changes_generic":"You have unsaved changes, are you sure you wish to leave?","add":"Add","remove":"Remove","select":"Select","no_followers_for_letter":"You don&rsquo;t follow anyone that begin with the letter &ldquo;{PAGE_LETTER}&rdquo;","share_limit_reached":"You have reached the maximum number of users to share with.","old_code":"Use old embed code","new_code":"Use new embed code","at_least_one":"There must be at least one user.","available":"Available","unavailable":"Unavailable","browse_error_generic":"Sorry, there was an error","browse_error_no_videos":"Sorry, no videos found","follow":"Follow","following":"Following","unfollow":"Unfollow","unfollowing":"Unfollowing","count_comments":{"singular":"{COUNT} comment","plural":"{COUNT} comments"},"first_comment":"Be the first to comment&hellip;","no_comments_for_you":"Forbidden. You cannot post comments on this page.","oops":"Oops!","player_try_again":"That wasn't supposed to happen. Please try again in a few minutes.","duration_input_min_duration":"The duration cannot be less than {MIN_DURATION}.","duration_input_max_duration":"The duration cannot be greater than {MAX_DURATION}.","duration_input_invalid_characters":"0-9 and : are the only acceptable inputs.","close":"Close","expand":"Expand","loading":"Loading...","top":"top","advanced_search":"Advanced Search","no_suggestions":"No suggestions","recent_searches":"Recent Searches","search_all":"Search All of Vimeo","clear_all_confirm":"Are you sure you want to clear your Watch Later videos?","totals":"Totals"}    };
