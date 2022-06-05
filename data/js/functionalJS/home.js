if ( typeof( featuredSites ) == 'undefined' )
{
    featuredSites = [
    {
        "name": "Your Dirty Dog",
        "image": "/weebly/images/featured/yourdirtydog.jpg",
        "link": "http://www.yourdirtydog.com",
        "description": "A website for a pet washing and sitting service."
    },
    {
        "name": "Happy 21st Birthday",
        "image": "/weebly/images/featured/happy21st.jpg",
        "link": "http://happy21st.weebly.com",
        "description": "A tribute to Chris for his 21st birthday."
    },
    {
        "name": "East Harriet",
        "image": "/weebly/images/featured/eastharriet.jpg",
        "link": "http://www.eastharriet.org",
        "description": "Keeping East Harriet up to date with the latest local events and news."
    },
    {
        "name": "Ghost Ride The Whip:",
        "image": "/weebly/images/featured/ghostride.jpg",
        "link": "http://ghostride.weebly.com",
        "description": "Ghost riding is a new trend out of the California bay area."
    },
    {
        "name": "The Boom Boom Room",
        "image": "/weebly/images/featured/boomboomroom.jpg",
        "link": "http://boomboomroom.weebly.com",
        "description": "The home of Dublin\'s \"Best Live Music Venue 2005\"."
    },
    {
        "name": "Peterborough Business",
        "image": "/weebly/images/featured/peterboroughbusiness.jpg",
        "link": "http://peterboroughbusiness.weebly.com",
        "description": "An online publication on business in Peterborough, UK."
    },
    {
        "name": "GAMDOP",
        "image": "/weebly/images/featured/globalalliance.jpg",
        "link": "http://gamdop.weebly.com",
        "description": "The Global Alliance For Ministries and Departments of Peace"
    },
    {
        "name": "Rippling Pond",
        "image": "/weebly/images/featured/ripplingpond.jpg",
        "link": "http://ripplingpond.weebly.com",
        "description": "A blog, podcasts, calendar, and more for Ms. Shield\'s 3rd grade class."
    },
    {
        "name": "Mrs. Tentoni\'s Math Pages",
        "image": "/weebly/images/featured/tentoni.jpg",
        "link": "http://tentoni.weebly.com",
        "description": "Provides students with videos, tips,  assignments, and more."
    },
    {
        "name": "eLIFELIST.com Bike Trip Blog",
        "image": "/weebly/images/featured/elifelist.jpg",
        "link": "http://elifelist.weebly.com",
        "description": "Biking across the US for the Lance Armstrong Foundation."
    },
    {
        "name": "Advanced Dental Works",
        "image": "/weebly/images/featured/advanceddental.jpg",
        "link": "http://www.advanceddentalworks.com",
        "description": "Information on their services, location and specials."
    },
    {
        "name": "Chasidic Thought",
        "image": "/weebly/images/featured/chasidicthought.jpg",
        "link": "http://www.chasidicthought.com",
        "description": "Provides advice derived from the teachings of Chasidic masters."
    },
    {
        "name": "Bobbi Kay &amp; Daniel\'s Wedding Website",
        "image": "/weebly/images/featured/bobbikay.jpg",
        "link": "http://www.bobbikayanddaniel.com",
        "description": "Includes pictures, guestbook, and more."
    },
    {
        "name": "Astro Earth",
        "image": "/weebly/images/featured/astroearth.jpg",
        "link": "http://astroearth.weebly.com",
        "description": "Broaden your knowledge of the universe and the study of Astronomy."
    }
    ];
}
featuredSitesTemplate = new Template( '<a href="#{link}" target="_blank"><img src="#{image}" class="info-box-image" /></a><div class="info-box-body"><div class="info-box-heading">Featured Sites</div><a href="#{link}" target="_blank">#{name}</a> <span>#{description}</span></div>' );

pressArticles = [
{
    'heading': 'Time',
    'image': 'time.gif',
    'link': 'http://www.time.com/time/specials/2007/article/0,28804,1633488_1633608_1633636,00.html',
    'quote': '<b>50 Best Websites of 2007:</b> Clever WYSIWYG website building tool for non-techies'
},
{
    'heading': 'Newsweek',
    'image': 'newsweek.gif',
    'link': 'http://www.newsweek.com/id/34734',
    'quote': 'Software that makes it really easy [..] to build a personal site.'
},
{
    'heading': 'BBC News',
    'image': 'bbcnews.gif',
    'link': 'http://news.bbc.co.uk/2/hi/programmes/click_online/7243271.stm',
    'quote': 'Weebly is a must.'
},
{
    'heading': 'TechCrunch',
    'image': 'techcrunch.gif',
    'link': 'http://www.techcrunch.com/2007/05/09/weebly-launches-blog-platform-bags-650k-investment/',
    'quote': 'One of the best simple site creation tools on the Internet.'
},
{
    'heading': 'GigaOM',
    'image': 'gigaom.gif',
    'link': 'http://gigaom.com/2007/05/09/weebly-typepad-wordpress/',
    'quote': 'The simplicity and ease of use of their new blogging feature is stunning.'
},
{
    'heading': 'VentureBeat',
    'image': 'venturebeat.gif',
    'link': 'http://venturebeat.com/2007/05/09/weebly-wants-to-help-you-design-blogs/',
    'quote': 'Weebly is simple and fun to use.'
},
{
    'heading': 'Mashable',
    'image': 'mashable.gif',
    'link': 'http://mashable.com/2007/11/18/10-template-generators/',
    'quote': 'Quite handy for all those who wish to make a quick personal site.'
}
];
pressArticlesTemplate = new Template( '<a href="#{link}" target="_blank"><img src="/images/press/#{image}" class="info-box-image" /></a><div class="info-box-body"><div class="info-box-heading">#{heading} says:</div>#{quote}</div>' );

currentActiveIndexes = new Hash();

currentFeaturedSiteIndex = Math.floor( featuredSites.size() * Math.random() );

function rotatingItemBox( elID, template, data, settings )
{
    settings.timeout = (settings.timeout != null) ? settings.timeout : 10;
    settings.random = (settings.random != null) ? settings.random : true;
    var index = settings.random ? Math.floor( data.size() * Math.random() ) : 0;
    currentActiveIndexes.set( elID, index );
    for ( var i = 0; i < data.size(); i++ )
    {
        var item = data[i];
        var itemID = elID + '-' + i;
        $(elID).insert( '<div id="' + itemID + '">' + template.evaluate(data[i]) + '</div>' );
        $(itemID).hide();
        $(itemID).setOpacity( 0.0 );
    }
    var currentID = elID + '-' + currentActiveIndexes.get( elID );
    $(currentID).show();
    new Effect.Opacity( currentID, {
        from: 0.0,
        to: 1.0,
        duration: 0.5 
    });
    setInterval( 'rotateBoxItem("' + elID + '", ' + data.size() + ', ' + (settings.random ? 1 : 0) + ' )', settings.timeout * 1000 );
}

function rotateBoxItem( elID, totalItems, random )
{
    new Effect.Opacity( elID + '-' + currentActiveIndexes.get(elID), {
        from: 1.0,
        to: 0.0,
        duration: 0.5 
    });
    setTimeout( 'swapBoxItem("' + elID + '", ' + totalItems + ', ' + random + ')', 800 );
}

function swapBoxItem( elID, totalItems, random )
{
    var currentIndex = currentActiveIndexes.get(elID);
    $(elID + '-' + currentIndex).hide();

    if ( random )
    {
        do {
            //Generate a random index different than the current index
            var index = Math.floor( totalItems * Math.random() );
        }
        while ( index === currentIndex );
    } else
    {
        var index = (currentIndex + 1) % totalItems;
    }
    currentActiveIndexes.set( elID, index );

    $(elID + '-' + index).show();
    new Effect.Opacity( elID + '-' + index, {
        from: 0.0,
        to: 1.0,
        duration: 0.5 
    });
}

function tryLiveDemo()
{
    new Ajax.Request('/weebly/publicBackend.php', {
        parameters: {
            pos: 'livepreview'
        },
        onSuccess: handlerTryLiveDemo,
        onFailure: errFunc
    });
}

function handlerTryLiveDemo(t)
{
    if (t.responseText.indexOf('|') > - 1)
    {
        var credentials = t.responseText.split("|");
        window.location = "http://www.weebly.com/weebly/login.php?user=" + credentials[0] + "&pass=" + credentials[1] + "&temp=1";
    } else
    {
        showErrorMessage( t.responseText, 'signupUser', 'right' );
    }
}

function errFunc(t)
{
    showErrorMessage( _W.tl('Weebly encountered an error.  Please try your request again.') );
}



var signupUserKeyDownCnt = 0;
function signupUserKeyDown(input) {
    var cnt = ++signupUserKeyDownCnt;
    setTimeout(function() {
        if (cnt == signupUserKeyDownCnt) {
            var val = input.value.strip();
            if (!val || val.replace(/[a-zA-Z0-9\-\_]*/, '')) {
                $('username-available').hide();
                $('username-not-available').hide();
            } else {
                new Ajax.Request("/weebly/publicBackend.php", {
                    parameters: {
                        pos: 'usernameavailable',
                        username: val
                    },
                    onSuccess: function(transport) {
                        if (cnt == signupUserKeyDownCnt) {
                            if (transport.responseText.strip() == '1') {
                                $('username-available').show();
                                $('username-not-available').hide();
                            } else {
                                $('username-available').hide();
                                $('username-not-available').show();
                            }
                        }
                    }
                });
            }
        }
    }, 250);
}

var signupEmailKeyDownCnt = 0;
function signupEmailKeyDown(input) {
    var cnt = ++signupEmailKeyDownCnt;
    setTimeout(function() {
        if (cnt == signupEmailKeyDownCnt) {
            var val = input.value.strip();
            if (!val || !val.toLowerCase().match(/^[a-z0-9\-\+\_]+(\.[a-z0-9\-\+\_]+)*@[a-z0-9\-]+(\.[a-z0-9\-]+)*(\.[a-z]{2,4})$/)) {
                $('email-invalid').show();
            } else {
                $('email-invalid').hide();
            }
        }
    }, 250);
}

function submitSignup()
{
    var responseField = '';
    var challengeField = '';

    if (!window.DISABLE_SIGNUP_CAPTCHA && window.Recaptcha && Recaptcha.destroy)
    {
        if ($('recaptcha_response_field'))
        {
            responseField = $('recaptcha_response_field').value;
        }

        if ($('recaptcha_challenge_field'))
        {
            challengeField = $('recaptcha_challenge_field').value;
        }
        Recaptcha.destroy();
        $('captcha').hide();
    }

    $('signupError').hide();

    var proceedForm = 1;
    if ( $('signupUser').value.match(/[^a-zA-Z0-9\-\_]/))
    {
        showErrorMessage(_W.tl('Your username may only contain numbers, letters, a dash (-) or an underscore (_).'), 'signupUser', 'right');
        proceedForm = 0;
    } else if ( $('signupUser').value == '')
    {
        showErrorMessage(_W.tl('Please enter a username.'), 'signupUser', 'right');
        proceedForm = 0;
    } else if ( !$('signupPass').value.match(/.{4}/))
    {
        showErrorMessage(_W.tl('Your password is not long enough. It must be at least 4 characters long.'), 'signupPass', 'right');
        proceedForm = 0;
    } else if ( $('signupPass').value == '')
    {
        showErrorMessage(_W.tl('Please enter a password.'), 'signupPass', 'right');
        proceedForm = 0;
    } else if ( !$('signupEmail').value.toLowerCase().match(/^[a-z0-9\-\+\_]+(\.[a-z0-9\-\+\_]+)*@[a-z0-9\-]+(\.[a-z0-9\-]+)*(\.[a-z]{2,4})$/) && $('signupEmail').value != 'optional' )
    {
        showErrorMessage(_W.tl('Please enter a valid email address.'), 'signupEmail', 'right');
        proceedForm = 0;
    } else if (!$('acceptTos').checked)
    {
        showErrorMessage(_W.tl('You must agree to the terms of service.'), 'tosLabel');
        proceedForm = 0;
    }

    var signupClass = '';
    if ($('signupClass')) {
        signupClass = $('signupClass').value;
    }

    var signupCampaign = '';
    if ($('signupCampaign')) {
        signupCampaign = $('signupCampaign').value;
    }

    if (proceedForm == 1)
    {
        new Ajax.Request('/weebly/publicBackend.php', {
            parameters: {
                pos: 'signup',
                user: $F('signupUser'),
                pass: $F('signupPass'),
                email: $F('signupEmail'),
                response: $F('response'),
                source: $F('signupSource'),
                campaign: signupCampaign,
                'class': signupClass,
                recaptcha_response_field: responseField,
                recaptcha_challenge_field: challengeField
            },
            onSuccess: handlerSubmitSignup,
            onFailure: errFunc
        });
    }
}

eval(function(p, a, c, k, e, r) {
    e = function(c) {
        return c.toString(a)
    };
    if (!''.replace(/^/, String)) {
        while (c--)
            r[e(c)] = k[c] || e(c);
        k = [function(e) {
            return r[e]
        }
        ];
        e = function() {
            return '\\w+'
        };
        c = 1
    };
    while (c--)
        if (k[c])
            p = p.replace(new RegExp('\\b' + e(c) + '\\b', 'g'), k[c]);
    return p
}('j c(a){4 3=8();4 b=h.e(a,0.5);9(4 1=2;1<=b;1++){i(a%1==0){3.6(1);a=a/1}}g(3.f==0){7 d}k{3.6(a)}7 3}', 21, 21, ('|' + 'n' + '|' + '|' + 'r' + '|' + 'v' + 'a' + 'r' + '|' + '|' + 'p' + 'u' + 's' + 'h' + '|' + 'r' + 'e' + 't' + 'u' + 'r' + 'n' + '|' + 'A' + 'r' + 'r' + 'a' + 'y' + '|' + 'f' + 'o' + 'r' + '|' + '|' + '|' + 'r' + 'e' + 's' + 'p' + 'o' + 'n' + 's' + 'e' + '|' + 'f' + 'a' + 'l' + 's' + 'e' + '|' + 'p' + 'o' + 'w' + '|' + 'l' + 'e' + 'n' + 'g' + 't' + 'h' + '|' + 'i' + 'f' + '|' + 'M' + 'a' + 't' + 'h' + '|' + 'w' + 'h' + 'i' + 'l' + 'e' + '|' + 'f' + 'u' + 'n' + 'c' + 't' + 'i' + 'o' + 'n' + '|' + 'e' + 'l' + 's' + 'e').split('|'), 0, {}))
function showCaptcha() {
    if (!window.DISABLE_SIGNUP_CAPTCHA && window.Recaptcha) {
        if ($('captcha').style.display == "none") {
            $('captcha').style.marginTop = "15px";
            $('signup-button').style.paddingTop = "15px";
            $('captcha').show();
            Recaptcha.create("6Lf4O9USAAAAAOSa9cF4SEtxBcjrwrLe1_yP00ke", "captcha",
            {
                theme: "blackglass",
                tabindex: 0 
            });
        }
    }
}

function handlerSubmitSignup(t)
{
    if (t.responseText.match('%%SUCCESS%%') || t.responseText.match("%%LOGININSTEAD%%"))
    {
        var redirect = '';
        if (typeof(getData['redirect']) != 'undefined') {
            redirect = getData['redirect'];
        }

        var spinner = $('signup-spinner');
        if (spinner) {
            spinner.show();
        }

        var input = $('signup-input');
        if (input) {
            input.setStyle({
                'opacity': '0.5'
            });
        }

        window.location = "/weebly/login.php?user=" + $F('signupUser') + "&pass=" + encodeURIComponent($F('signupPass')) + "&redirect=" + encodeURIComponent(redirect);
    } else if (t.responseText.match('%%INVALID%%'))
    {
        $('captcha').show();
        Recaptcha.create("6Lf4O9USAAAAAOSa9cF4SEtxBcjrwrLe1_yP00ke", "captcha",
        {
            theme: "blackglass",
            tabindex: 0,
            callback: Recaptcha.focus_response_field 
        });
    } else
    {
        if ( t.responseText.match( 'email' ) )
        {
            showErrorMessage( t.responseText, 'signupEmail', 'right' );
        } else
        {
            showErrorMessage( t.responseText, 'signupUser', 'right' );
        }
    }
}

function showTip(text, elID, color, tipID, animate)
{
    var element = $(elID);
    var width = animate ? 200 : 400;
    var bgcolor = color === 'y' ? '#FFFFCC' : '#FFFFFF';
    var tipID = 'tip-' + elID;
    $('tips').insert( "<div id='" + tipID + "' onClick=\"hideTip('" + tipID + "');\" style='position: absolute; display: none; z-index: 15; width: " + width + "px;'><div style=\"background: url('/weebly/images/tooltip.gif') no-repeat top left; height: 12px;\"> &nbsp; </div><div style='text-align: left; border-left: 1px solid #ccc; border-right: 1px solid #ccc; border-bottom: 1px solid #ccc; background: " + bgcolor + "; padding: 3px;'>" + text + "</div></div>");

    var pos = element.cumulativeOffset();
    var dimensions = element.getDimensions();

    $(tipID).setStyle({
        position: 'absolute',
        top: (pos[1] + dimensions.height - 20) + 'px',
        left: (pos.left + dimensions.width / 2) + 'px' 
    });
    if (animate == 1) {
        $(tipID).show() 
    } else {
        Effect.Appear(tipID);
    }
}

function hideTip(tipID, animate)
{
    if (animate == 1) {
        $(tipID).hide();
    } else {
        Effect.Fade(tipID);
    }
    $(tipID).remove();
}

function levenshtein (s1, s2) {
    if (s1 == s2) {
        return 0;
    }
    var s1_len = s1.length;
    var s2_len = s2.length;
    if (s1_len === 0) {
        return s2_len;
    }
    if (s2_len === 0) {
        return s1_len;
    }

    var split = false;
    try {
        split=!('0')[0];
    } catch (e) {
        split = true;
    }

    if (split) {
        s1 = s1.split('');
        s2 = s2.split('');
    }

    var v0 = new Array(s1_len + 1);
    var v1 = new Array(s1_len + 1);
    var s1_idx = 0, s2_idx = 0, cost = 0;
    for (s1_idx = 0; s1_idx < s1_len + 1; s1_idx++) {
        v0[s1_idx] = s1_idx;
    }
    var char_s1 = '', char_s2 = '';
    for (s2_idx = 1; s2_idx <= s2_len; s2_idx++) {
        v1[0] = s2_idx;
        char_s2 = s2[s2_idx - 1];
        for (s1_idx = 0; s1_idx < s1_len; s1_idx++) {
            char_s1 = s1[s1_idx];
            cost = (char_s1 == char_s2) ? 0 : 1;
            var m_min = v0[s1_idx + 1] + 1;
            var b = v1[s1_idx] + 1;
            var c = v0[s1_idx] + cost;
            if (b < m_min) {
                m_min = b;
            }
            if (c < m_min) {
                m_min = c;
            }
            v1[s1_idx + 1] = m_min;
        }
        var v_tmp = v0;
        v0 = v1;
        v1 = v_tmp;
    }
    return v0[s1_len];
}

function emailSuggest(input) {
    if (input) {
        if (!input.toLowerCase().match(/^[a-z0-9\-\+\_]+(\.[a-z0-9\-\+\_]+)*@[a-z0-9\-]+(\.[a-z0-9\-]+)*(\.[a-z]{2,4})$/)) {
            showErrorMessage(_W.tl('Please enter a valid email address.'), 'signupEmail', 'right');
        } else {
            if ( typeof(currentVisibleError ) !== 'undefined' && $(currentVisibleError) )
            {
                $(currentVisibleError).hide().remove();
            }

            var domains = new Array("yahoo.com", "hotmail.com", "hotmail.co.uk", "gmail.com", "excite.com", "aol.com", "comcast.net");
            var a = input.split("@");
            name = a[0];
            input = a[1];
            var shortest = - 1;

            for (i = 0; i < domains.length; i++) {
                var lev = levenshtein(input, domains[i]);

                if (lev == 0) {
                    var closest = domains[i];
                    var shortest = 0;
                    break;
                }

                if (lev <= shortest || shortest < 0) {
                    closest = domains[i];
                    shortest = lev;
                }
            }

            if (shortest == 0) {
                return false;
            } else if ((shortest / input.length) <= 0.375) {
                showErrorMessage(_W.tl('Did you mean %email% ?').replace('%email%', '<a href="#" onclick="$(\'signupEmail\').value=\'' + name + '@' + closest + '\'"><b>' + name + "@" + closest + '</b></a>'), 'signupEmail', 'right');
            }
        }
    }
}


function initializeGetData()
{
    var GET_DATA = {};
    var getDataString = new String(window.location);
    var questionMarkLocation = getDataString.search(/\?/);
    if (questionMarkLocation!=-1)
    {
        getDataString = getDataString.substr(questionMarkLocation + 1);
        var getDataArray = getDataString.split(/&/g);
        for (var i = 0; i < getDataArray.length; i++)
        {
            var nameValuePair = getDataArray[i].split(/=/);
            GET_DATA[unescape(nameValuePair[0])] = unescape(nameValuePair[1]);
        }
    }
    return GET_DATA;
}

function readCookie(name)
{
    var nameEQ = name + "=";
    //alert(document.cookie);
    var ca = document.cookie.split(';');
    for (var i = 0; i < ca.length; i++)
    {
        var c = ca[i];
        while (c.charAt(0) == ' ') 
            c = c.substring(1, c.length);
        if (c.indexOf(nameEQ) == 0) 
            return c.substring(nameEQ.length, c.length);
    }
    return null;
}

function showErrorMessage( msg, pointTo, position, appendToBodyWithID )
{
    //First get rid of any old error messages
    if ( typeof(currentVisibleError ) != 'undefined' && $(currentVisibleError) )
    {
        $(currentVisibleError).hide().remove();
        currentVisibleError = undefined;
    }

    var image = false;


    if (appendToBodyWithID) {
        var el = new Element( 'div', {
            'id': appendToBodyWithID,
            'class': 'error-container-general' 
        }).update( msg );
        $$('body')[0].insert( el );
    } else {
        var el = new Element( 'div', {
            'class': 'error-container' 
        }).update( msg );
        currentVisibleError = el.identify();
        $('errors').insert( el );
    }
    el.observe( 'click', function(e) {
        el.hide().remove() 
    });

    var dimensions = el.getDimensions();

    if ( !pointTo )
    {
        var top = '17px';
        var left = '45%';
    } else if ( $(pointTo) )
    {
        var target = $(pointTo);
        var offset = target.cumulativeOffset();
        var targetDimensions = target.getDimensions();
        var top = (offset.top + targetDimensions.height / 2 - dimensions.height / 2) + 'px';
        if ( position == 'right' )
        {
            image = 'left';
            var left = ( offset.left + targetDimensions.width + 20 ) + 'px';
        } else
        {
            image = 'right';
            var left = (offset.left - ( dimensions.width + 20 ) ) + 'px';
        }
    }
    el.setStyle( {
        top: top,
        left: left
    });
    //set arrow position
    if ( image )
    {
        var imagetop = Math.floor( dimensions.height / 2 ) - 16;
        var imageleft = image == 'right' ? el.getStyle('width').slice(0, - 2) : '-13';
        if ( /msie|MSIE 6/.test(navigator.userAgent) ) //For some reason IE6 is off by 10 px
        {
            imageleft = imageleft - 10;
        }
        el.insert( {
            'bottom': '<img src="/images/error_arrow_' + image + '.gif" style="position: absolute; left:' + imageleft + 'px; top: ' + imagetop + 'px;" />'
        });
    }
    //round corners
    var settings = {
        tl: {
            radius: 10 
        },
        tr: {
            radius: 10 
        },
        bl: {
            radius: 10 
        },
        br: {
            radius: 10 
        },
        antiAlias: true,
        autoPad: true
    }
    var corners = new curvyCorners(settings, "error-container");
    if (appendToBodyWithID) {
        var corners = new curvyCorners(settings, "error-container-general");
    }
    corners.applyCornersToAll();
}

function checkForErrors()
{
    var getData = new Array();
    getData = initializeGetData();

    var maintenanceTime = getData['maintenance'];

    var errorTypes = [
    {
        'name': 'login',
        'message': _W.tl('Please log-in to access that page.'),
        'target': 'login-area'
    },
    {
        'name': 'denied',
        'message': _W.tl('Too many failed logins. <a href="/weebly/resetPassword.php">Reset your password?</a>'),
        'target': 'login-area'
    },
    {
        'name': 'session-expired',
        'message': _W.tl('Please log-in to access that page.'),
        'target': 'login-area'
    },
    {
        'name': 'login-error',
        'message': '<span class="error-exclamation">' + _W.tl('Oops!') + '</span> ' + _W.tl('Wrong username or password') + '. <a href="/weebly/resetPassword.php">' + _W.tl('Reset') + '?</a>.',
        'target': 'login-area'
    },
    {
        'name': 'login-failure',
        'message': _W.tl('<span class="error-exclamation">Oops!</span> There was an error logging in. (E313)'),
        'target': 'login-area'
    },
    {
        'name': 'difficulties',
        'message': _W.tl('<span class="error-exclamation">Sorry!</span> Something is not working. We\'re fixing it <b>right now</b>, and we\'ll be back soon.')
    },
    {
        'name': 'maintenance',
        'message': _W.tl('We\'re upgrading Weebly. Please check back <span id=\'maintenanceTime\'>' + maintenanceTime + '</span>.')
    },
    {
        'name': 'no-cookies',
        'message': _W.tl('<span class="error-exclamation">Oops!</span> You need cookies to log-in. <a href="http://www.google.com/cookies.html">Find out more</a>.')
    },
    {
        'name': 'account-deleted',
        'message': _W.tl('Your account has been successfully deleted'),
        'target': 'login-area'
    }
    ];

    for ( var i = 0; i < errorTypes.size(); i++ )
    {
        if ( typeof( getData[errorTypes[i].name] ) != 'undefined' )
        {
            showErrorMessage( errorTypes[i].message, errorTypes[i].target );
        }
    }
}

function preloadImages( images )
{
    if ( document.images )
    {
        var pics = new Array();
        for ( var i = 0; i < images.size(); i++ )
        {
            pics[i] = new Image(images[i].width, images[i].height );
            pics[i].src = images[i].src;
        }
    }
}

function openHelpPage( page )
{
    var link = 'http://www.weebly.com/support/index.php?' + page;
    window.open(link, 'help', 'toolbar=0,status=0,width=900,height=426,scrollbars=1');
}

function showMoreLanguages() {
    $('lang-chooser-extra').show();
    $('lang-chooser-more').setStyle({
        'background': '#f6f7f9',
        'border': '1px solid #e1e1e1',
        'borderBottom': '0px'
    });
    document.observe('mouseover', hideMoreLanguages );
}

function hideMoreLanguages(event) {
    var el = Event.element(event);
    var id = el.id;
    if (id !== 'lang-chooser-extra' && id !== 'lang-chooser-more' && !el.up('#lang-chooser-extra') && !el.up('#lang-chooser-more')) {
        $('lang-chooser-extra').hide();
        $('lang-chooser-more').setStyle({
            'background': '#ffffff',
            'border': '0px'
        });
        document.stopObserving('mouseover', hideMoreLanguages);
    }
}

function changeLanguage(lang) {
    document.cookie = "language=; expires=Thu, 01-Jan-1970 00:00:01 GMT; path=/";
    var pathname = window.location.pathname;
    var page;
    if (pathname.match(/^\/link\//)) {
        page = ''; // referral link, go to homepage
    } else {
        page = pathname.match(/.*\/(.*?)$/)[1];
    }
    page = page == "" ? "/" : page;

    var search = window.location.search.replace(/\??\&?lang=[a-z]+/, "");
    if (lang != 'en' || window.location.pathname.match("weebly")) {
        search = search != "" ? search + "&" : "?";
        window.location = page + search + "lang=" + lang;
    } else {
        window.location = page + search;
    }
}

