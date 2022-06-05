/**
 * Provides a mixin for data collections
 *
 * <p><b>Require Path:</b> shared/data/helpers/collection-mixin</p>
 *
 * @module Shared
 * @submodule Shared.Account
 * @namespace Account
 * @static AccountMixin
**/
define('shared/data/helpers/collection-mixin',[
    'jquery/nyt',
    'underscore/nyt'
], function ($, _) {
    'use strict';

    var DataCollectionMixin = {

        hasFetched: false,

        /**
         * Return the next model in the collection
         *
         * @private
         * @method getNext
         * @return next model
        **/
        getNext: function (feed, model) {
            var index = feed.indexOf(model);
            var next = index === feed.length ? 0 : index + 1;

            return feed.at(next);
        },

        /**
         * Fetches the collection if it hasn't been fetched yet
         *
         * @private
         * @method isFetchingData
         * @return {Boolean} A determination of whether the data is being fetched.
        **/
        loadData: function () {
            if (this.length === 0 && !this.hasFetched) {
                this.hasFetched = true;
                this.fetch();
            }

            return this;
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
            return 'rref=' + this.sectionId;
        },

        /**
         * Returns the Name of the collection
         *
         * @public
         * @method getName
         * @return {String} The name of the collection
        **/
        getName: function () {
            return this.sectionName;
        },

        /**
         * Returns the Section Front Url of the collection
         *
         * @public
         * @method getUrl
         * @return {String} The section front url of the collection
        **/
        getUrl: function () {
            return this.sectionUrl;
        }
    };

    return DataCollectionMixin;

});

