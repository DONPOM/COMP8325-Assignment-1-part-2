/**
 * Creates a User Data Model that is used to gain information about the logged in user.
 *
 * <p><b>Require Path:</b> foundation/models/user-data</p>
 *
 * @module Foundation
 * @class UserData
 * @static
 * @extends Backbone.Model
 **/
define('foundation/models/user-data',[
    'jquery/nyt',
    'underscore/nyt',
    'backbone/nyt',
    'foundation/models/base-model',
    'foundation/helpers/model-mixin',
    'foundation/cookies',
    'foundation/hosts'
], function ($, _, Backbone, BaseModel, modelMixin, Cookie, hosts) {
    'use strict';

    var UserDataModel = BaseModel.extend(modelMixin).extend({

        defaults: {
            id: '',
            name: '',
            profileImage: '',
            subscription: []
        },

        eventName: 'nyt:user-ready',

        /**
         * Fetchs the data when we create our user.
         *
         * @private
         * @method constructor
         **/
        initialize: function () {
            var userData = this;

            //when synced, check if the user has a profile image and change our ready state
            this.subscribeOnce(this, 'sync error', this.loadProfileImage);
            this.subscribeOnce(this, 'sync error', function () {
                userData.hasSynced = true;
                userData.broadcast(this.eventName);
            });

            this.listenTo(this.pageManager, 'nyt:nytnow-enable', this.handleAddingNow);

            //load the user data as soon as we instanciate the model
            this.loadData();
        },

        /**
         * Decides whether to load the user data via the Varnish ESI or a fetch to the server
         *
         * @private
         * @method loadData
         **/
        loadData: function () {
            var user = $('#userinfo').text();

            if (user) {
                this.set($.parseJSON(user).data);
                this.trigger('sync');
            } else {
                this.fetch();
            }
        },

        /**
         * Called when we want to fetch the data from our cookie cracking service
         *
         * @private
         * @method sync
         * @param method {String} Create, Read, Update, or Delete
         * @param model {Object} the model to be saved (or collection to be read)
         * @param options {Object} jquery ajax request options
         * @return {Object} Backbone Sync
         **/
        sync: function (method, model, options) {
            options.url = hosts.www + '/svc/web-products/userinfo-v2.json';
            options.cache = false;

            //use jsonp for non-www domains
            if (hosts.www !== ('//' + window.location.host)) {
                options.dataType = 'jsonp';
                options.jsonpCallback = 'userInfoCallback';

                options.url += 'p';
                options.data = {
                    cookie: Cookie.readCookie('NYT-S') || ''
                };
            }

            return Backbone.sync(method, model, options);
        },

        /**
         * Called when we want to fetch the data from our cookie cracking service
         *
         * @private
         * @method parse
         * @return {Object} Modified user data json object
         **/
        parse: function (response) {
            return response.data;
        },

        /**
         * Marketing campaigns can fire events that add the NOW entitlement
         *
         * @private
         * @method handleAddingNow
        **/
        handleAddingNow: function () {
            this.get('subscription').push('NOW');
        },

        /**
         * Determines if the user has a specific entitlement
         *
         * @public
         * @method hasEntitlement
         * @param entitlementName {String} The name of the entitlement
         * @return {Boolean}
         **/
        hasEntitlement: function (name) {
            return _.indexOf(this.get('subscription'), name) > -1;
        },

        /**
         * Determines if the user is logged in
         *
         * @public
         * @method isLoggedIn
         * @return {Boolean}
         **/
        isLoggedIn: function () {
            var id = this.get('id');
            return id.length > 0 && id !== '0';
        },

        /**
         * Get the id of the user logged in
         *
         * @public
         * @method getUserId
         * @return {String}
         **/
        getUserId: function () {
            return this.get('id');
        },

        /**
         * Get the name of the user logged in
         *
         * @public
         * @method getUserName
         * @return {String}
         **/
        getUserName: function () {
            return this.get('name');
        },

        /**
         * Get the truncated name of the user logged in
         *
         * @public
         * @method getTruncatedUserName
         * @param {Number} truncationLimit Integer representing the character limit
         * @return {String} Truncated username
        **/
        getTruncatedUserName: function (truncationLimit) {
            var username = this.get('name');

            if (_.isNumber(truncationLimit) && username.length > truncationLimit) {
                username = username.substring(0, truncationLimit - 1) + '...';
            }

            return username;
        },

        /**
         * Get the profile image of the user logged in if he or she has one
         *
         * @public
         * @method getImageUrl
         * @return {String}
         **/
        getImageUrl: function () {
            return this.get('profileImage');
        },

        /**
         * Determines if the user has a subscription for the web
         *
         * @public
         * @method isWebSubscriber
         * @return {Boolean}
         **/
        isWebSubscriber: function () {
            return this.hasEntitlement('MM');
        },

        /**
         * Determines if the user has a subscription for mobile devices
         *
         * @public
         * @method isMobileSubscriber
         * @return {Boolean}
         **/
        isMobileSubscriber: function () {
            return this.hasEntitlement('MTD') && this.hasEntitlement('MSD');
        },

        /**
         * Determines if the user has a subscription for tablets
         *
         * @public
         * @method isTabletSubscriber
         * @return {Boolean}
         **/
        isTabletSubscriber: function () {
            return this.hasEntitlement('MTD');
        },

        /**
         * Determines if the user has a subscription for smartphones
         *
         * @public
         * @method isSmartphoneSubscriber
         * @return {Boolean}
         **/
        isSmartphoneSubscriber: function () {
            return this.hasEntitlement('MSD');
        },

        /**
         * Determines if the user has a subscription for Crosswords
         *
         * @public
         * @method isCrosswordsSubscriber
         * @return {Boolean}
         **/
        isCrosswordsSubscriber: function () {
            return this.hasEntitlement('XWD');
        },

        /**
         * Determines if the user has a subscription to the print
         *
         * @public
         * @method isHomeDeliverySubscriber
         * @return {Boolean}
         **/
        isHomeDeliverySubscriber: function () {
            return this.hasEntitlement('HD');
        },

        /**
         * Determines if the user has a subscription to NYT Now
         *
         * @public
         * @method isNytNowSubscriber
         * @return {Boolean}
         **/
        isNytNowSubscriber: function () {
            return this.hasEntitlement('NOW');
        },

        /**
         * Determines if the user has a subscription to cooking
         *
         * @public
         * @method isCookingSubscriber
         * @return {Boolean}
         **/
        isCookingSubscriber: function () {
            return this.hasEntitlement('COO');
        },

        /**
         * Determines if the user has a subscription to opinion
         *
         * @public
         * @method isOpinionSubscriber
         * @return {Boolean}
         **/
        isOpinionSubscriber: function () {
            return this.hasEntitlement('OPI');
        },

        /**
         * Determines if the user has a subscription to Times Premier
         *
         * @public
         * @method isPremierSubscriber
         * @return {Boolean}
         **/
        isPremierSubscriber: function () {
            return this.hasEntitlement('TPR');
        },

        getImageSource: function(id) {
            var src = hosts.profileImage + '/' + id.substring(4, 0) + '/' + id.substring(4) + '/cropped-' + id + '.jpg';
            return src;
        },

        /**
         * Tries to load a user's profile image
         *
         * @private
         * @method loadProfileImage
         **/
        loadProfileImage: function () {
            if (!this.isLoggedIn()) {
                return;
            }

            var user = this;
            var profileImage = new Image();
            var id = user.getUserId();
            var src = user.getImageSource(id);

            if (id.length > 0) {
                profileImage.onload = function () {
                    user.set('profileImage', src);
                    /**
                     * Fired when a new profile image is loaded.
                     * @event nyt:user-image-loaded
                    **/
                    user.local(user, 'nyt:user-image-loaded');
                };

                profileImage.src = src;
            }
        }
    });

    return new UserDataModel();
});

