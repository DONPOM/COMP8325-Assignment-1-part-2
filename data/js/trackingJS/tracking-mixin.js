/**
* The base mixin for adding tracking functionality to views
*
* <p><b>Require path:</b> foundation/tracking/tracking-mixin</p>
*
* @module Foundation
* @class TrackingMixin
*/
define('foundation/tracking/tracking-mixin',[
    'jquery/nyt',
    'underscore/nyt',
    'foundation/helpers/utils',
    'foundation/views/page-manager',
    'foundation/hosts',
    'foundation/cookies'
], function ($, _, utils, pageManager, hosts, cookies) {
    'use strict';

    var TrackingMixin = {

        /**
        * Default object mixed into every view so that if a view has
        *  no common, base data, it doesn't have to define it explicitly.
        *
        * @type object
        */
        trackingBaseData: {},

        /**
        * Utility method to combine base tracking data with the provided object
        *
        * @method trackingCombineData
        * @param {Object} data to combine or override with base data
        * @return {Object}
        */
        trackingCombineData: function (obj) {
            obj = obj || {};
            try {
                return _.extend({}, this.trackingBaseData, obj, {'pgtype': this.pageManager.getMeta('PT')});
            }
            catch(e) {
                //just in case
            }
        },

        /**
        * Utility method to send an object to the TAGX event proxy
        *  - automatically combines the base data before sending to the 
        *    wrapped 'raw' trigger function
        *
        * @method trackingTrigger
        * @param {String} identifying string for TAGX handling
        * @param {Object} data to send
        */
        trackingTrigger: function (action, trackingData) {
            trackingData = this.trackingCombineData(trackingData);
            this.trackingTriggerRaw(action, trackingData, 'interaction');
        },

        /**
        * Utility method to send an object to the TAGX event proxy
        *  - automatically combines the base data before sending to the 
        *    wrapped 'raw' trigger function. sets the type value to impression
        *
        * @method trackingTriggerImpression
        * @param {String} identifying string for TAGX handling
        * @param {Object} data to send
        */
        trackingTriggerImpression: function (action, trackingData) {
            trackingData = this.trackingCombineData(trackingData);
            this.trackingTriggerRaw(action, trackingData, 'impression');
        },

        /**
        * Utility method to send an object to the TAGX event proxy
        *
        * @method trackingTriggerRaw
        * @param {String} identifying string for TAGX handling
        * @param {Object} data to send
        */
        trackingTriggerRaw: function (action, trackingData, type) {
            var success = pageManager.trackingTriggerEvent(action, trackingData, type);
            if (!success) {
                pageManager.trackingAddToEventQueue({
                    action: action,
                    data: trackingData,
                    type: type
                });
            }
        },

        /**
        * Utility method to convert an object into a query string for appending to URLs
        *
        * @method trackingAppendParams
        * @param {String} href the href value from the link
        * @param {Object} paramsObj the object to serialize into a string for the URL
        * @return {String} the new url with the meta data appended to it
        */
        trackingAppendParams: function (href, paramsObj) {
            var retUrl;

            //NB: tracking with cookies removed
            //add the custom cookie to the tracking parameters
            //paramsObj.t = cookieName;
            //trackingParamsStr = paramsList.join('&');
            //trackingParamsStr = this.writeTrackingParamsCookie(trackingParamsStr);
            //return cookieFailed ? href + separator + trackingParamsStr : href + separator + 't=' + cookieName;

            retUrl = utils.appendQueryParams(href, this.trackingCombineData(paramsObj));

            return retUrl;
        },

        /**
         * Writes provided tracking parameters to a cookie
         *
         * @private
         * @method writeTrackingParamsCookie
         * @param {String} trackingParamsStr Tracking parameters to write to cookie
         * @return {String} Unchanged or modified tracking parameters string
        **/
        writeTrackingParamsCookie: function (trackingParamsStr) {
            var cookieFailed;
            var cookieExp = new Date();
            var cookieName = 'qry' + Math.ceil(Math.random() * 1000);

            // cookie expires in 2 min
            cookieExp.setMinutes(cookieExp.getMinutes() + 2);

            // try to set the cookie
            cookies.writeCookie(cookieName, trackingParamsStr, { expires: cookieExp });

            //failed cookies get an extra paramater for tracking purposes.
            if (cookies.readCookie(cookieName) === null) {
                trackingParamsStr = 'alxcookie=0&' + trackingParamsStr;
            }

            return trackingParamsStr;
        },

        trackingComscorePVC: function (data) {
            new Image().src = hosts.comscorePvc + '/svc/comscore/pvc.html';
            if (data) {
                this.trackingTriggerRaw('comscore-pageview-candidate', data, 'other');
            }
        }
    };

    return TrackingMixin;
});


