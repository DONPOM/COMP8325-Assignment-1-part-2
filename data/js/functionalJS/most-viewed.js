/**
 * Creates a new instance of the most viewed collection
 *
 * <p><b>Require Path:</b> shared/data/instances/most-viewed</p>
 *
 * @module Shared
 * @submodule Shared.Data
 * @class MostViewedInstance
 * @static
**/
define('shared/data/instances/most-viewed',[
    'jquery/nyt',
    'foundation/views/page-manager',
    'foundation/hosts',
    'shared/data/collections/most-popular'
], function ($, pageManager, Hosts, MostPopular) {
    'use strict';

    return new MostPopular([], {
        type: 'mostviewed'
    });

});

