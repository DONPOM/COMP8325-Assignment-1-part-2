/**
* Utility helper with common component-independent functionality
*
* <p><b>Require path:</b> foundation/helpers/utils</p>
*
* @module Foundation
* @class Utils
*/
define('foundation/helpers/utils',[
    'foundation/views/page-manager',
    'foundation/hosts',
    'foundation/browser'
], function (pageManager, hosts, browser) {
    'use strict';

    var Utils = {

        /**
         * Appends parameters to a provided url
         *
         * @public
         * @method appendQueryParams
         * @param {String} url Url to be manipulated
         * @param {Object} params Dictionary of values to be appended to url
         * @return {String} Manipulated url with appended query parameters
        **/
        appendQueryParams: function (url, params) {
            var key, separator, query;
            var paramsList = [];
            var anchor = document.createElement('a');

            params = params || {};

            anchor.href = url;
            query = anchor.search;

            for (key in params) {
                var paramPart = encodeURIComponent(key);

                paramPart += (params[key] !== '') ? '=' : '';
                paramsList.push(paramPart + encodeURIComponent(params[key]));
                if (query.indexOf(paramPart) !== -1) {
                    query = query.replace(new RegExp('&?' + paramPart + '[^&]*'), '');
                }
            }

            query = (query.length === 1 && query.indexOf('?') === 0) ? '' : query;
            separator = (query.length > 0 && query.indexOf('?') === 0) ? '&' : '?';

            url = [
                anchor.protocol !== ':' ? anchor.protocol + '//' : 'http://',
                anchor.hostname !== '' ? anchor.hostname : '',
                '/', anchor.pathname.replace(/^\//, ''),
                query, separator, paramsList.join('&'),
                anchor.hash
            ].join('');

            return url;
        },

        /**
         * Returns a host (intl vs www) based on the current edition
         *
         * @public
         * @method getHostBasedOnEdition
         * @return {String}
        **/
        getHostBasedOnEdition: function () {
            if (pageManager.isInternationalEdition() && browser.getWindow().location.host.indexOf('international') >= 0) {
                return hosts.international;
            }

            return hosts.www;
        }

    };

    return Utils;
});

