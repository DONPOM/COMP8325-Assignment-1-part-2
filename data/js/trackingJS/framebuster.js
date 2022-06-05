/**
 * Busts out of iframes for sites that are not white listed.
 *
 * <p><b>Require Path:</b> foundation/framebuster</p>
 *
 * @module Foundation
 * @class Framebuster
 * @static
**/
define('foundation/framebuster',[
    'foundation/cookies',
    'foundation/browser'
], function(Cookies, Browser) {
    'use strict';

    var expTime;
    var referrer = Browser.getDocument().referrer;
    var top = Browser.getWindow().top;
    var location = Browser.getWindow().location;

    var frameRegExp = /^https?:\/\/(?:[^?\/]+\.)?(?:localhost|(?:nytimes|nytlabs|stumbleupon|starbucks)\.com|(?:newsdev\.net))(?:\:[1-9][0-9]*)?\//;

    if (
        window.self !== top &&
        !(location.pathname.match(/^\/portals/) && location.hostname === 'www.nytimes.com') && //bug in starbucks requires portals
        !referrer.match(frameRegExp) &&
        !top.location.href.match(frameRegExp)
    ) {
        //Set a cookie expiration time of 1 minute
        expTime = new Date();
        expTime.setTime(expTime.getTime() + 60000);

        //Write temporary Cookie with referrer
        Cookies.writeCookie({
            name: 'FramesetReferrer',
            value: referrer,
            options: {
                expires: new Date(expTime)
            }
        });

        //Redirect to iframed page.
        top.location.replace(location.pathname);
    }

    //Return the Regex for Unit Testing
    return frameRegExp;
});

