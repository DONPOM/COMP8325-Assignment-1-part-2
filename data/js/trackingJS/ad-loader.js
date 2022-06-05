window.NYTADX = {};

/**
 * AdX positions for the article page
 *
 * @public
 * @param NYTADX[positions]
**/
window.NYTADX.positions = [
    'Interstitial',
    'Header1',
    'Middle1C',
    'TopLeft',
    'TopRight',
    'Top5',
    'Top',
    'Top_Close',
    'Box1',
    'Bar1',
    'MOTHAd',
    'Middle',
    'HPmodule-RE2',
    'Middle4',
    'Middle5',
    'HPMiddle',
    'HPMiddle3',
    'HPPromo',
    'HPBottom1',
    'HPMidLeader',
    'HPBreak',
    'HPSponLink',
    'HPWeather',
    'Bottom8',
    'Bottom9',
    'Anchor',
    'Inv1',
    'Inv2',
    'Inv3',
    'ab1',
    'ab2',
    'ab3',
    'prop1',
    'prop2'
];

/**
 * Constructs an ADX call and adds a blocking script tag
 *
 * @public
 * @namespace NYTADX
 * @method buildAdx
**/
window.NYTADX.buildAdx = function (keywords, sensitivity, isBlocking) {
    'use strict';

    //adx host
    var host = 'www.nytimes.com';
    if (/sbx|dev|stg/.test(window.location.host)) {
        host = 'www.stg.nytimes.com';
    }

    /**
     * Provides a way to access the url parameters
     *
     * @private
     * @method getParams
    **/
    var getParams = (function () {
        var i, param, len, urlParams;
        var urlString = window.location.search;
        var params = {};

        if (urlString.length === 0) {
            return false;
        }

        urlParams = urlString.substring(1).split('&');

        for (i = 0, len = urlParams.length; i < len; i += 1) {
            param = urlParams[i].split('=');
            if (param[1] && param[1].indexOf(',') >= 0) {
                params[param[0]] = param[1].split(',');
            } else {
                params[param[0]] = param[1] || '';
            }
        }

        return params;
    }());

    /**
     * Returns the campaign page with a query string override
     *
     * @private
     * @method getPage
    **/
    var getPage = function () {
        var page = getParams.adx_campaign;

        //if no adx_campaign param, assign default homepage
        if (!getParams.adx_campaign) {

            //edit the campaign if an ad sensitivity occurs
            if (sensitivity) {
                page = 'www.nytimes.com/' + sensitivity;
            } else {
                page = (window.location.host.indexOf('international') > -1) ? 'international-home' : 'homepage';
                page += '.nytimes.com/index.html';
            }

        }

        return page;
    };

    /**
     * Returns the positions that should be included in the ad call
     *
     * @private
     * @method getPositions
    **/
    var getPositions = function () {
        var exclusions, i, j;

        // include only these positions when using ?adx_positions
        if (getParams.adx_positions) {
            return getParams.adx_positions;
        
        // exclude only these positions when using ?adx_exclude_positions
        } else if (getParams.adx_exclude_positions) {
            exclusions = getParams.adx_exclude_positions;
            
            for (i = 0; i < window.NYTADX.positions.length; i += 1) {
                for (j = 0; j < exclusions.length; j += 1) {
                    if (window.NYTADX.positions[i] === exclusions[j]) {
                        window.NYTADX.positions.splice(i, 1);
                    }
                }
            }

            return window.NYTADX.positions.join(',');
        
        //all other cases use the default
        } else {
            return window.NYTADX.positions.join(',');
        }
    };

    /**
     * Returns the keywords supplied by the query string and the article
     *
     * @private
     * @method getKeywords
    **/
    var getKeywords = function () {
        keywords.push('nyt5');

        //add special homepage refresh logic
        if (getParams['WT.z_jog'] === '1') {
            keywords.push('rf');
        }

        //add any keywords from the url
        if (getParams['ad-keywords']) {
            keywords.push(getParams['ad-keywords']);
        }

        return encodeURIComponent(keywords.join(','));
    };

    /**
     * Returns the adx settings
     *
     * @private
     * @method getAdxSettings
    **/
    var getAdxSettings = function () {
        var param;
        var deliveryProofParams = ['adxforce', 'debug'];
        var settings = [
            'autoconfirm=0',
            'interstitials=1',
            'cpp=1',
            'v=3',
            'attributes=nyt5'
        ];

        // proof of ad delivery
        for (var i = deliveryProofParams.length - 1; i >= 0; i -= 1) {
            param = getParams[deliveryProofParams[i]];

            if (typeof param !== 'undefined') {
                settings.push(deliveryProofParams[i] + '=' + (param || '1'));
            }
        }

        return settings.join('&');
    };

    /**
     * Creates the URL to send off to ADX
     *
     * @private
     * @method constructUrl
    **/
    var constructUrl = function () {
        return 'http://' + host + '/adx/bin/adxrun.html?jsonp=NYTADX.processAdx' +
            '&page=' + getPage() +
            '&positions=' + getPositions() +
            '&' + getAdxSettings() +
            '&keywords=' + getKeywords();
    };

    //If AdX is running well, block execution of the page. Otherwise fall back to
    //Backbone's async approach.
    if (isBlocking === true || typeof isBlocking === 'undefined') {
        document.write('<scr' + 'ipt src="' + constructUrl() + '"></sc' + 'ript>');
    }

};

/**
 * Determines if ad object provided is blank or not
 *
 * @private
 * @namespace NYTADX
 * @method isBlank
 * @param {Object} ad Ad object from adxrun response
**/
window.NYTADX.isBlank = function (ad) {
    'use strict';

    var isBlank = false;

    ad = ad || {};

    if (typeof ad.classification === 'undefined') {
        isBlank = true;
    } else if (ad.classification === 'blank' || ad.classification === 'blank-but-count-imps') {
        isBlank = true;
    }

    return isBlank;
};

/**
 * Processes the JSONP response from ADX
 *
 * @public
 * @namespace NYTADX
 * @method processAdx
**/
window.NYTADX.processAdx = function (response) {
    'use strict';

    var rootEl = document.getElementsByTagName('html')[0];
    var ads = response.ads || {};
    var isBlank = window.NYTADX.isBlank;

    //Determines if adx returned any ads
    var hasAds = function () {
        var ads = response.ads || {};

        if (Object.keys) {
            return Object.keys(ads).length !== 0;
        } else {
            var prop;
            var keyCount = 0;

            //count the ads old school style
            for (prop in ads) {
                keyCount+=1;
            }

            return keyCount !== 0;
        }
    };

    //Add class if there is an Interstitial ad served
    if (ads.Interstitial && !isBlank(ads.Interstitial)) {
        rootEl.className += ' has-homepage-interstitial-ad';
    }

    //Add class if there is a Middle4 ad served
    if (ads.Middle4 && !isBlank(ads.Middle4)) {
        rootEl.className += ' has-middle4-ad';
    }

    //Add class if a long c-column ad exists
    if ((ads.HPMiddle && !isBlank(ads.HPMiddle)) || (ads.Middle && !isBlank(ads.Middle))) {
        rootEl.className += ' has-tall-c-column-ad ';
    }

    //Add class if there is a MOTH ad served
    if (ads.MOTHAd && !isBlank(ads.MOTHAd)) {
        rootEl.className += ' has-moth-ad';
    }

    //When there are no ads, add a special class
    if (!hasAds()) {
        rootEl.className += ' no-ads';
    }

    //only add the ads if there is a response
    if (ads) {
        window.NYTADX.response = response;
    }
};

/**
 * Paints an ad on the page immediately if available in the response
 *
 * @public
 * @namespace NYTADX
 * @method paintAd
**/
window.NYTADX.paintAd = function (name) {
    var ad, confirmImage, confirmUrl;
    var adx = window.NYTADX;

    if (adx && adx.response && adx.response.ads && adx.response.ads[name]) {
        ad = adx.response.ads[name];
        confirmUrl = ad['confirmation-url'];

        //inject the ad into the page
        document.getElementById(name).innerHTML = ad.creative;

        //confirm the ad
        if (confirmUrl && confirmUrl !== '') {
            confirmImage = new Image();
            confirmImage.src = confirmUrl;
        }

        //inform the ad view to ignore this position since we already painted it
        window.NYTADX.response.ads[name].hasPainted = true;
    }
};

/**
 * Sets the dimensions on the page so there is no janky experience
 *
 * @public
 * @namespace NYTADX
 * @method setDimensions
**/
window.NYTADX.setDimensions = function (name) {
    var ad, height;
    var el = document.getElementById(name);
    var adx = window.NYTADX;

    if (el && adx && adx.response && adx.response.ads && adx.response.ads[name]) {
        ad = adx.response.ads[name];
        height = ad['height'];

        //set the height
        if (height) {
            el.className = el.className.replace(/hidden/, '');
            el.style.minHeight = height + 'px';
        }
    }
};

/**
 * Builds interstitial and makes requirements for ad functioning
 *
 * @public
 * @namespace NYTADX
 * @method buildInterstitialAd
**/
window.NYTADX.buildInterstitialAd = function() {
    'use strict';

    var adContainer = document.getElementById('InterstitialContainer');

    if (!window.NYTADX.response || !window.NYTADX.response.ads || !window.NYTADX.response.ads.Interstitial) {
        return;
    }

    adContainer.className = adContainer.className.replace('hidden', '');

    require(
        ['foundation/main'],
        function () {
            require(
                ['backbone/nyt', 'jquery/nyt', 'foundation/views/ad-view-manager'],
                window.NYTADX.renderInterstitialAd,
                window.NYTADX.destroyInterstitialAd
            );
        },
        window.NYTADX.destroyInterstitialAd
    );
};

/**
 * Renders interstitial ad
 *
 * @public
 * @namespace NYTADX
 * @method renderInterstitialAd
 * @param {Object} Backbone reference to Backbone
 * @param {Object} AdViewManager reference to foundational ad view manager
**/
window.NYTADX.renderInterstitialAd = function(Backbone, $, AdViewManager) {
    'use strict';

    var timeout;
    var ad = window.NYTADX.response.ads.Interstitial;
    var timeoutDuration = (ad.timeout || 15) * 1000;
    var close = function() {
        window.clearTimeout(timeout);
        window.NYTADX.destroyInterstitialAd();
        AdViewManager.broadcast('nyt:check-ads-force');
    };

    AdViewManager.render({
        el: $('#Interstitial'),
        model: new Backbone.Model(ad),
        frameStyle: '<style type="text/css">body { margin: 0; overflow: hidden; } a img { border: none; }</style>',
        callback: function() {
            $('#InterstitialContainer > button').on('click', close);
        }
    });

    timeout = window.setTimeout(close, timeoutDuration);
};

/**
 * Destroys interstitial ad
 *
 * @public
 * @namespace NYTADX
 * @method destroyInterstitialAd
 */
window.NYTADX.destroyInterstitialAd = function() {
    'use strict';

    var rootElement = document.getElementsByTagName('html')[0];
    var adContainer = document.getElementById('InterstitialContainer');

    adContainer.className += ' hidden';
    adContainer.innerHTML = '';

    rootElement.className = rootElement.className.replace('has-homepage-interstitial-ad', '');
};

/**
 * Inativity timer-related module for publicly accessible start and stop methods
 *
 * @public
 * @namespace NYTADX
 * @module inactivityTimer
**/
window.NYTADX.inactivityTimer = {
    start: function () {},
    stop: function () {},
    init: function () {
        'use strict';

        var inactivityTimer = this;

        if (typeof require === 'undefined') {
            return;
        }

        require(['foundation/main'], function () {
            require(['foundation/views/page-manager'], function (pageManager) {
                var moduleName = 'ad-loader';

                inactivityTimer.start = function () {
                    pageManager.trigger('nyt:inactivity-timer-enable', moduleName);
                };

                inactivityTimer.stop = function () {
                    pageManager.trigger('nyt:inactivity-timer-disable', moduleName);
                };
            });
        });
    }
};
