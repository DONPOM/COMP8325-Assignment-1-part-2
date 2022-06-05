/**
 * Creates a new collection for user recommendations.
 *
 * <p><b>Require Path:</b> shared/data/collections/recommendations</p>
 *
 * @module Shared
 * @submodule Shared.Data
 * @class Recommendations
 * @constructor
 * @extends BaseCollection
**/
define('shared/data/collections/recommendations',[
    'backbone/nyt',
    'foundation/collections/base-collection',
    'shared/data/models/article',
    'foundation/hosts',
    'underscore/nyt',
    'shared/data/helpers/collection-mixin'
], function (Backbone, BaseCollection, Article, hosts, _, collectionMixin) {
    'use strict';


    var Recommendations = BaseCollection.extend(

        _.extend({}, collectionMixin, {

            model: Article,

            url: hosts.du + '/svc/recommendations/v3/personalized.jsonp',

            /**
             * Called each time we want to fetch a new set of articles from a particular feed
             *
             * @private
             * @method sync
             * @param method {String} Create, Read, Update, or Delete
             * @param model {Object} the model to be saved (or collection to be read)
             * @param options {Object} jquery ajax request options
             * @return {Object} Backbone Sync
            **/
            sync: function (method, model, options) {
                options.dataType = 'jsonp';
                options.jsonpCallback = 'jsonRecommended';
                options.jsonp = 'callback';
                options.cache = true;
                options.data = {
                    access_key: '776ACB1E-6C7F-4702-9E18-CF329376F5A3'
                };
                return Backbone.sync(method, model, options);
            },

            /**
             * Modifies the format of the recommended collection results
             *
             * @private
             * @method parse
             * @param response {Object} The data returned after a sync is completed
             * @return {Object} Revised JSON response to be added to the collection
            **/
            parse: function (response) {
                var collection = this;
                var responseItems = _.map(response.suggestions, function(responseItem){
                    var media = responseItem.thumbs || {};

                    //map promotional media to match scoop output
                    responseItem.promotional_media = {
                        image: {
                            image_crops: {
                                thumbStandard: responseItem.thumbnail,
                                mediumThreeByTwo210: collection.getCrop(media, 'mediumThreeByTwo210')
                            }
                        }
                    };

                    // Add video label to kicker
                    if (responseItem.kicker && responseItem.item_type === 'Video') {
                        responseItem.kicker = 'Video' + ' | ' + responseItem.kicker;
                    } else if (responseItem.item_type === 'Video') {
                        responseItem.kicker = 'Video';
                    }

                    // Check if the bylines are not empty.
                    if (responseItem.byline) {
                        responseItem.authors = [responseItem.byline.replace('By ', '').replace(/\w\S*/g, function(txt){return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();})];
                    } else {
                        responseItem.authors = '';
                    }

                    //map data fields to match output in scoop
                    responseItem.description = responseItem.abstract;
                    responseItem.link = responseItem.url;
                    responseItem.headline = responseItem.title;
                    responseItem.pubDate = responseItem.published_date;
                    return responseItem;
                });

                this.num_articles = response.num_articles;
                this.uid = response.uid;
                this.user_displayname = response.user_displayname;
                this.user_pic_url = response.user_pic_url;

                return responseItems;
            },

            getCount: function () {
                return this.num_articles || 0;
            },

            getUserId: function () {
                return this.uid || 0;
            },

            getUserDisplayName: function () {
                return this.user_displayname || null;
            },

            getUserPictureUrl: function () {
                return this.user_pic_url || null;
            },

            /**
             * Returns a specific crop to be used in most emailed
             *
             * @private
             * @method getCrop
             * @param data {Object} An array of crops
             * @param name {Object} The crop name to pluck
             * @return {Object} The data cooresponding with the appropriate crop.
            **/
            getCrop: function (data, name) {
                var i, len;

                for (i = 0, len = data.length; i < len; i += 1) {
                    if (data[i].type === name) {
                        data[i].url = data[i].content;
                        return data[i];
                    }
                }
            },

            /**
             * Returns the query parameter that is used to identify what to load on the
             * next page view
             *
             * @private
             * @method getIdentifier
             * @return {String} A query parameter string
            **/
            getIdentifier: function () {
                return 'src=recg';
            }
        })
    );

    return Recommendations;
});

