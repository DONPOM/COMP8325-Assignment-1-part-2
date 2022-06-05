/**
 * Creates a new instance of the most emailed collection
 *
 * <p><b>Require Path:</b> shared/data/instances/most-emailed</p>
 *
 * @module Shared
 * @submodule Shared.Data
 * @class MostEmailedInstance
 * @static
**/
define('shared/data/instances/most-emailed',[
    'jquery/nyt',
    'foundation/views/page-manager',
    'foundation/hosts',
    'shared/data/collections/most-popular'
], function ($, pageManager, Hosts, MostPopular) {
    'use strict';

    return new MostPopular([], {
        type: 'mostemailed'
    });

});
