/**
 * Creates a new instance of the weather collection
 *
 * <p><b>Require Path:</b> shared/data/instances/weather-collection</p>
 *
 * @module Shared
 * @submodule Shared.Data
 * @class WeatherCollectionInstance
 * @static
**/
define('shared/data/instances/weather-collection',[
    'shared/data/collections/weather'
], function (WeatherCollection) {
    'use strict';

    return new WeatherCollection();
});

