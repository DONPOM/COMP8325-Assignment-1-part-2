/**
 * Creates a collection of most popular articles
 *
 * <p><b>Require Path:</b> shared/data/collections/most-popular</p>
 *
 * @module Shared
 * @submodule Shared.Data
 * @namespace Data.MostPopular
 * @class Collection
 * @constructor
 * @extends BaseCollection
**/
define('shared/data/collections/most-popular',[
    'backbone/nyt',
    'foundation/collections/base-collection',
    'shared/data/models/article',
    'foundation/hosts',
    'underscore/nyt',
    'shared/data/helpers/collection-mixin'
], function (Backbone, BaseCollection, Article, Hosts, _, collectionMixin) {
    'use strict';


    var MostPopular = BaseCollection.extend(

        _.extend({}, collectionMixin, {

            model: Article,

            apis: {
                mostviewed: {
                    url: Hosts.s1 + '/du/mostpopular/viewed/all_1.jsonp?v1',
                    callback: 'jsonCallbackViewed'
                },
                mostemailed: {
                    url: Hosts.s1 + '/du/mostpopular/emailed/all_1.jsonp?v1',
                    callback: 'jsonCallbackEmailed'
                }
            },

            initialize: function(models, options) {
                this.url = this.apis[options.type].url;
                this.callback = this.apis[options.type].callback;
            },

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
                options.jsonpCallback = this.callback;
                options.jsonp = 'callback';
                options.cache = true;
                return Backbone.sync(method, model, options);
            },

            /**
             * Modifies the format of the article collection results
             *
             * @private
             * @method parse
             * @param response {Object} The data returned after a sync is completed
             * @return {Object} Revised JSON response to be added to the collection
            **/
            parse: function (response) {
                var collection = this;
                var responseItems = _.map(response.results, function(responseItem){
                    var media = [];
                    if(responseItem.id) {
                        responseItem.data_id = responseItem.id;
                        delete responseItem.id;
                    }
                    if (responseItem.asset_id) {
                        delete responseItem.asset_id;
                    }
                    if (responseItem.media === '') {
                        delete responseItem.media;
                    } else {
                        media = responseItem.media[0]['media-metadata'] || [];
                    }

                    //map promotional media to match scoop output
                    responseItem.promotional_media = {
                        image: {
                            image_crops: {
                                thumbStandard: collection.getCrop(media, 'Standard Thumbnail'),
                                mediumThreeByTwo210: collection.getCrop(media, 'mediumThreeByTwo210')
                            }
                        }
                    };

                    //map data fields to match output in scoop
                    responseItem.description = responseItem.abstract;
                    responseItem.link = responseItem.url;
                    responseItem.kicker = responseItem.column;
                    responseItem.headline = responseItem.title;
                    responseItem.authors = [responseItem.byline.replace('By ', '').replace(/\w\S*/g, function(txt){return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();})];
                    responseItem.pubDate = responseItem.published_date;
                    return responseItem;
                });

                return responseItems;
            },

            /**
             * Returns a specific crop to be used in most popular list
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
                    if (data[i].format === name) {
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
                return 'src=me';
            }
        })
    );

    return MostPopular;
});

