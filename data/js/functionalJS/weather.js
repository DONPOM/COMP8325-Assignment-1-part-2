/**
 * Creates a collection for weather data.
 *
 * <p><b>Require Path:</b> shared/data/collections/weather</p>
 *
 * @module Shared
 * @submodule Shared.Data
 * @class WeatherCollection
 * @constructor
 * @extends BaseCollection
**/
define('shared/data/collections/weather',[
    'backbone/nyt',
    'foundation/collections/base-collection',
    'underscore/nyt',
    'shared/data/helpers/collection-mixin',
    'foundation/helpers/utils',
    'shared/data/models/weather'
], function (Backbone, BaseCollection, _, collectionMixin, utils, Weather) {
    'use strict';

    var WeatherCollection = BaseCollection.extend(

        _.extend({}, collectionMixin, {

            model: Weather,
            url: '',

            initialize: function (options) {
                this.settings = _.extend({}, this.defaultSettings, options);
            },

            /**
             * Overriding mixin helper function. Called by the weather view to populate collection with weather forecast data
             *
             * @private
             * @method loadData
            **/
            loadData: function(location) {
                this.url = utils.getHostBasedOnEdition() + '/svc/weather/v2/current-and-five-day-forecast.jsonp';
                this.fetch();
            },

            /**
             * Called each time we want to fetch weather forecast data from DU Weather Forecast API
             *
             * @private
             * @method sync
             * @param method {String} Create, Read, Update, or Delete
             * @param model {Object} the model to be saved (or collection to be read)
             * @param options {Object} jquery ajax request options
             * @return {Object} Backbone Sync
            **/
            sync: function (method, model, options) {
                options.url = this.url;
                options.dataType = 'jsonp';
                options.jsonpCallback = 'jsonWeather';
                options.jsonp = 'callback';
                options.cache = true;
                return Backbone.sync(method, model, options);
            },

            /**
             * Modifies the format of the weather collection results
             *
             * @private
             * @method parse
             * @param response {Object} The data returned after a sync is completed
             * @return {Object} Revised JSON response to be added to the collection
            **/
            parse: function (response) {
                var forecastWeather, currentWeather;
                var currentWeatherAttributes = ['description', 'temp', 'temp_unit', 'icon'];

                if (response && response.results) {
                    forecastWeather = response.results.five_day_forecast;
                    // weather for current day
                    currentWeather = response.results.current[0];
                    // add in the needed fields from current weather to corresponding day in forecast
                    for (var attribute in currentWeatherAttributes) {
                        if (currentWeatherAttributes.hasOwnProperty(attribute)) {
                            forecastWeather[0][currentWeatherAttributes[attribute]] = currentWeather[currentWeatherAttributes[attribute]];
                        }
                    }
                    return forecastWeather;
                } else {
                    return [];
                }
            },

            /**
             * Returns the weather models for the four day weather forecast
             *
             * @public
             * @method getFourDayForecast
             * @return {Array} Weather forecast
            **/
            getFourDayForecast: function () {
                return _.rest(this.models, 1);
            },

            /**
             * Returns the current weather model
             *
             * @public
             * @method getCurrentWeather
             * @return {Object} Current weather
            **/
            getCurrentWeather: function () {
                return this.models[0];
            }
        })
    );

    return WeatherCollection;
});

