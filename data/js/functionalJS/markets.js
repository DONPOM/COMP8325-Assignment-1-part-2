/**
 * Creates a new collection for markets on demand
 *
 * <p><b>Require Path:</b> shared/data/collections/markets</p>
 *
 * @module Shared
 * @submodule Shared.Data
 * @class Markets
 * @constructor
 * @extends BaseCollection
**/
define('shared/data/collections/markets',[
    'backbone/nyt',
    'foundation/collections/base-collection',
    'shared/data/models/markets',
    'foundation/hosts',
    'underscore/nyt'
], function (Backbone, BaseCollection, MarketsModel, hosts, _) {
    'use strict';

    var Markets = BaseCollection.extend({

        /**
         * Initialize and fetch the markets data
         *
         * @private
         * @method constructor
        **/
        initialize: function () {
            this.fetch();
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
            options.url = 'http://markets.on.nytimes.com/research/Net/SectionFrontAPI/MarketModuleData/jsonp';
            options.dataType = 'jsonp';
            options.jsonpCallback = 'jsonMarkets';
            options.jsonp = 'callback';
            options.cache = true;
            return Backbone.sync(method, model, options);
        },

        /**
         * Modifies the format of the markets on demand collection results
         *
         * @private
         * @method parse
         * @param response {Object} The data returned after a sync is completed
         * @return {Object} Revised JSON response to be added to the collection
        **/
        parse: function (response) {
            var i, j, regLen, idxLen, index;
            var regions = response.Tabs;

            //massage the data before the backbone collection consumes it
            for (j = 0, regLen = regions.length; j < regLen; j += 1) {

                //Keep the first three markets only
                if (regions[j].Items.length > 3) {
                    regions[j].Items = regions[j].Items.slice(0, 3);
                }

                //Create a new model for each index
                for (i = 0, idxLen = regions[j].Items.length; i < idxLen; i += 1) {
                    regions[j].Items[i] = new MarketsModel(regions[j].Items[i]);
                }

            }

            return regions || [];
        },

        /**
         * Add commas to large dollar amounts
         *
         * @method addNumberCommas
        **/
        addNumberCommas: function (val) {
            while (/(\d+)(\d{3})/.test(val.toString())){
                val = val.toString().replace(/(\d+)(\d{3})/, '$1'+','+'$2');
            }
            return val;
        },

        /**
         * Locates the index that is open and caches the decision locally
         *
         * @private
         * @method determineOpenIndex
        **/
        determineOpenIndex: function () {
            var i, j, len, region, index;
            var indexes = ['U.S.', 'Europe', 'Asia'];

            //iterate through regions to see which markets are open until one is found
            for (i = 0; i < 3 && !this.openRegion; i += 1) {

                //cache region for easy access
                region = this.findWhere({Label: indexes[i]});

                //iterate through indexes to check to see if it's open
                for (j = 0, len = region.get('Items').length; j < len; j += 1) {

                    //cache the index for a specific region
                    index = region.get('Items')[j];

                    if (index.get('IsOpen') === 'True') {
                        this.openRegion = region;
                        break;
                    }
                }

            }

            //if no markets are open, default with the U.S.
            this.openRegion = this.openRegion || this.findWhere({Label: 'U.S.'});

            return this.openRegion;
        },

        /**
         * Get the index that is open
         *
         * @public
         * @method getOpenIndex
        **/
        getOpenIndex: function () {
            return this.openRegion || this.determineOpenIndex();
        },

        /**
         * Get a specific index
         *
         * @public
         * @method getIndex
        **/
        getIndex: function (index) {
            return this.findWhere({Label: index});
        }

    });

    return Markets;
});

