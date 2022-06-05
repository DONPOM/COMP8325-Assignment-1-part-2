/**
 * Provides an API to make working with browser cookies a bit easier.
 *
 * <p><b>Require Path:</b> foundation/cookies</p>
 *
 * @module Foundation
 * @class Cookies
 * @static
**/
define('foundation/cookies',[
    'jquery/nyt'
], function ($) {
    'use strict';

    var defaultOptions = {
        expires: 365,
        domain: '.nytimes.com',
        path: '/',
        secure: false
    };

    /**
    * Reads the stored information for a particular cookie
    *
    * @method readCookie
    * @param name {String} the name of a cookie you would like to get the contents of
    */
    var readCookie = function (name) {
        var i, len, cookieParts, cookieName;
        var cookiesList = document.cookie.split('; ');
        for (i = 0, len = cookiesList.length; i < len; ++i) {
            cookieParts = cookiesList[i].split('=');
            cookieName = cookieParts.shift();
            if (cookieName === name) {
                return cookieParts.join('=');
            }
        }
        return null;
    };

    /**
    * Writes a new cookie
    *
    * @method writeCookie
    * @param name {String} The name of a cookie you would like to write
    * @param value {String} The value of the cookie data
    * @param options {Object} Optional options
    * @param options.domain {String} The domain in which the Cookie is available
    * @param options.path {String} The Path in which the Cookie is available
    * @param options.expires {Integer} The value in days
    */
    var writeCookie = function (name, value, options) {
        var days, now;

        options = $.extend({}, defaultOptions, options);

        if (value === null) {
            options.expires = -1;
        }

        if ($.isNumeric(options.expires)) {
            days = options.expires;
            now = new Date();
            now.setDate(now.getDate() + days);
            options.expires = now;
        }

        document.cookie = [
            name, '=', value,
            '; expires=' + options.expires.toUTCString(),
            '; path=' + options.path,
            '; domain=' + options.domain,
            options.secure ? '; secure' : ''
        ].join('');
    };

    /**
    * Deletes a cookie
    *
    * @method deleteCookie
    * @param name {String} The name of a cookie you would like to delete
    */
    var deleteCookie = function (name, options) {
        var cookieVal = readCookie(name);
        if (cookieVal !== null) {
            writeCookie(name, null, options);
        }
    };

    var cookiesEnabled = function () {
        var name = "NYTCookiesEnabled";
        var exp = new Date();
        exp.setTime(exp.getTime() + 10000);
        document.cookie = name + "=test; expires=" + exp.toGMTString();
        var result = (document.cookie.indexOf(name) !== -1);
        if (result) {
            // only expire if the cookie was actually set
            exp = new Date();
            exp.setTime(exp.getTime() - 10000);
            document.cookie = name + "=test; expires=" + exp.toGMTString();
        }
        return result;
    };

    return {
        readCookie: readCookie,
        writeCookie: writeCookie,
        deleteCookie: deleteCookie,
        cookiesEnabled: cookiesEnabled
    };

});

