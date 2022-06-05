/**
 * Creates page storage module that interacts with local storage
 *
 * <p><b>Require Path:</b> foundation/models/page-storage</p>
 *
 * @module Foundation
 * @class UserData
 * @static
 * @extends Backbone.Model
 **/
define('foundation/models/page-storage',[
    'jquery/nyt',
    'underscore/nyt',
    'backbone/nyt',
    'foundation/models/base-model',
    'foundation/helpers/model-mixin',
    'foundation/hosts'
], function ($, _, Backbone, BaseModel, modelMixin, hosts) {
    'use strict';

    var PageStorage = BaseModel.extend(modelMixin).extend({

        /**
         * @param storageKey {Object} Local storage key
        **/
        storageKey: 'nyt:site-storage',

        /**
         * @param serviceUrl {Object} Url to the local storage page
        **/
        serviceUrl: hosts.www + '/svc/web/localstorage.html',

        /**
         * @param serviceUrl {Object} Url to the local storage page
        **/
        eventName: 'nyt:storage-ready',

        /**
         * Fetchs the the local storage data on initialize
         *
         * @private
         * @method constructor
        **/
        initialize: function () {
            var storageData;

            _.bindAll(this, 'handleFrameLoad', 'handleMessage');

            //Cross domain calls open up an iframe to www
            if (this.isXDomain()) {
                this.setupXDomain();
            } else {
                this.syncData(window.localStorage.getItem(this.storageKey) || '{}');
            }
        },

        /**
         * Helper method to support backwards compatibility. Use set instead.
         *
         * @private
         * @deprecated
         * @method saveOnChange
        **/
        syncData: function (jsonString) {
            try {
                this.clear().set(JSON.parse(jsonString));
            } catch (e) {
                // Invalid JSON
            }

            this.hasSynced = true;
            this.broadcast(this.eventName);
            this.listenTo(this, 'change', this.handleDataChange);
        },

        /**
         * Saves the JSON data to local storage whenever there is a change to the model.
         *
         * @private
         * @method saveOnChange
        **/
        handleDataChange: function () {
            var data = JSON.stringify(this.toJSON());

            if (this.isXDomain()) {
                this.sendMessage(data);
            } else {
                window.localStorage.setItem(this.storageKey, data);
            }
        },

        /**
         * Determines whether to use xdomain localstorage
         *
         * @private
         * @method isSameDomain
        **/
        isXDomain: function () {
            return window.location.host.indexOf('www') < 0;
        },


        // ***************** CROSS DOMAIN FUNCTIONALITY *****************

        /**
         * Creates an iFrame and sets up the events for xDomain use
         *
         * @private
         * @method setupXDomain
        **/
        setupXDomain: function () {
            // Creates an iframe for communicating cross domain
            this.iframe = document.createElement("iframe");
            this.iframe.style.cssText = "position:absolute;width:1px;height:1px;left:-9999px;";
            document.body.appendChild(this.iframe);

            // Modern Browsers: Post Message
            if (window.addEventListener){
                this.iframe.addEventListener("load", this.handleFrameLoad, false);
                window.addEventListener("message", this.handleMessage, false);

            // Old IE: Post Message
            } else if (this.iframe.attachEvent){
                this.iframe.attachEvent("onload", this.handleFrameLoad, false);
                window.attachEvent("onmessage", this.handleMessage);
            }

            this.iframe.src = this.serviceUrl;
        },

        /**
         * Sends a message to the iframe. If there is no value sent, the entire
         * localstorage package will be returned.
         *
         * @private
         * @method sendMessage
        **/
        sendMessage: function (value) {
            var data = {key: this.storageKey};

            if (value) {
                data.value = value;
            }

            this.iframe.contentWindow.postMessage(JSON.stringify(data), window.location.protocol + hosts.www);
        },

        /**
         * Requests the local storage data when the iframe is loaded.
         *
         * @private
         * @method handleFrameLoad
        **/
        handleFrameLoad: function (e) {
            this.sendMessage();
        },

        /**
         * Retreives the post message from the iframe.
         *
         * NOTE: The email this app uses post message - to remain consistent, data is
         * separated with colons: e.g. 'localstorage:{"foo":"bar"}'
         *
         * @private
         * @method handleMessage
        **/
        handleMessage: function (e) {
            var message = e.data;
            if (/\.nytimes\.com$/.test(e.origin) && message.indexOf('localstorage:') === 0) {
                this.syncData(message.replace('localstorage:', ''));
            }
        }
    });

    if (!window.Modernizr.localstorage || !window.Modernizr.postmessage) {
        PageStorage = Backbone.Model.extend({
            ready: $.noop
        });
    }

    return new PageStorage();
});

