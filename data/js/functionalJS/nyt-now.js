/**
 * Creates a new collection for nyt now articles.
 *
 * <p><b>Require Path:</b> shared/data/collections/nyt-now</p>
 *
 * @module Shared
 * @submodule Shared.Data
 * @class NytNow
 * @constructor
 * @extends BaseCollection
**/
define('shared/data/collections/nyt-now',[
    'backbone/nyt',
    'foundation/collections/base-collection',
    'foundation/helpers/utils',
    'underscore/nyt',
    'shared/data/helpers/collection-mixin'
], function (Backbone, BaseCollection, utils, _, collectionMixin) {
    'use strict';


    var NytNow = BaseCollection.extend(

        _.extend({}, collectionMixin, {

            url: utils.getHostBasedOnEdition() + '/services/json/channels/nytnow.json',

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
                options.dataType = 'json';
                options.cache = true;
                return Backbone.sync(method, model, options);
            },

            /**
             * Modifies the format of the nyt now collection results
             *
             * @private
             * @method parse
             * @param response {Object} The data returned after a sync is completed
             * @return {Object} Revised JSON response to be added to the collection
            **/
            parse: function (response) {

                var responseItems = _.map(response.NytNow.urls, function(responseItem){
                    var obj = {'url' : responseItem};
                    return obj;
                });

                return responseItems;
            }
        })
    );

    return NytNow;
});

