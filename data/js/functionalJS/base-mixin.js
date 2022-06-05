/**
 * Creates a base mixin so we can implement common functionality across our bases
 *
 * <p><b>Require Path:</b> foundation/base-mixin</p>
 *
 * @module Foundation
 * @static Base-Mixin
**/
define('foundation/base-mixin',[
    'jquery/nyt',
    'underscore/nyt',
    'foundation/views/page-manager'
], function ($, _, pageManager) {
    'use strict';

    var base = {

        /**
         * @param pageManager {Object} Adding the pageManager to every view.
        **/
        pageManager: pageManager,

        dateHelper: {
            months: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],

            /**
             * Returns the name of a month given a zero-indexed offset
             *
             * @public
             * @method monthName
             * @param {Integer} Zero-indexed month offset
             */
            getMonthName: function(monthOffset) {
                return this.months[monthOffset];
            }
        },

        /**
         * Returns if a feature flag is on or off
         *
         * @public
         * @method flag
         * @param flagName {String} The name of the feature flag
         * @return {Boolean}
        **/
        flag: function (flagName) {
            return pageManager.flag(flagName);
        },

        /**
         * Trigger events on the page manager.
         *
         * @public
         * @method broadcast
         * @param eventName {String} The name of the event to be triggered
         * @param args {Mixed} Information that is sent with the trigger.
        **/
        broadcast: function (eventName, args) {
            pageManager.trigger.apply(pageManager, arguments);
        },

        /**
         * Locally broadcast an event on the provided view.
         *
         * @public
         * @method broadcastLocal
         * @param view {Object} The local view in which you are broadcasting on
         * @param eventName {String} The name of the event to be triggered
         * @param args {Mixed} Information that is sent with the trigger.
        **/
        local: function (view, eventName, args) {
            var localArgs = Array.prototype.slice.call(arguments, 0);
            localArgs.shift();
            view.trigger.apply(view, localArgs);
        },

        /**
         * Subscribe to events broadcasted on the page manager
         *
         * @public
         * @method subscribe
         * @param [view] {Object} An optional backbone view
         * @param eventName {String} The name of the event to be triggered
         * @param callback {Mixed} The callback to be triggered
        **/
        subscribe: function () {
            this.subscribeHelper('listenTo', arguments);
        },

        /**
         * Subscribe a single event broadcasted on the page manager
         *
         * @public
         * @method subscribeOnce
         * @param [view] {Object} An optional backbone view
         * @param eventName {String} The name of the event to be triggered
         * @param callback {Mixed} The callback to be triggered
        **/
        subscribeOnce: function () {
            this.subscribeHelper('listenToOnce', arguments);
        },

        /**
         * Stop subscribing to a specific event broadcasted on the page manager
         *
         * @public
         * @method stopSubscribing
         * @param [view] {Object} An optional backbone view
         * @param eventName {String} The name of the event to be triggered
         * @param callback {Mixed} The callback to be triggered
        **/
        stopSubscribing: function () {
            this.subscribeHelper('stopListening', arguments);
        },

        /**
         * Helper function to build subscription functions
         *
         * @private
         * @method subscribeHelper
         * @param [view] {Object} An optional backbone view
         * @param eventName {String} The name of the event to be triggered
         * @param callback {Mixed} The callback to be triggered
        **/
        subscribeHelper: function (type, args) {
            if (_.isString(args[0])) {
                this[type](pageManager, args[0], args[1]);
            } else {
                this[type](args[0], args[1], args[2]);
            }
        },

        /**
         * Creates an anchor for a provided url
         *
         * @public
         * @method createAnchor
         * @param url {String} an object similar to window.location
        **/
        createAnchor: function (url) {
            var anchor = document.createElement('a');
            anchor.href = url;
            return anchor;
        },

        /**
         * Wrapper function exposing isUserVariant
         *
         * @public
         * @method isUserVariant
         * @param query {String} The name or id of the test to query against
         * @return {Boolean}
        **/
        isUserVariant: function (query) {
            return window.NYTABTEST.engine.isUserVariant(query);
        },

        /**
         * Wrapper function exposing isUserControl
         *
         * @public
         * @method isUserControl
         * @return {Boolean}
        **/
        isUserControl: function (query) {
            return window.NYTABTEST.engine.isUserControl(query);
        },

        /**
         * Wrapper function exposing loadGoogleAnalytics
         *
         * @public
         * @method loadGoogleAnalytics
        **/
        loadGoogleAnalytics: function () {
            var deferred = new $.Deferred();

            if (window.NYTABTEST.engine.isUserThrottle()) {
                require(['foundation/lib/ga/ga-nyt5'], function(gaNyt5) {
                    deferred.resolve(gaNyt5);
                });
            }

            return deferred.promise();
        }
    };

    return base;

});

