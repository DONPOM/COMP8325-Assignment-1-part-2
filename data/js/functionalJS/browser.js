/**
 * Wraps browser elements so that they can be mocked out for testing
 *
 * <p><b>Require Path:</b> foundation/browser</p>
 *
 * @module Foundation
 * @class Browser
 * @static
 **/
define('foundation/browser',[], function() {
    'use strict';
    var Browser = {
        getDocument: function () {
            return document;
        },

        getWindow: function () {
            return window;
        }
    };

    return Browser;
});
