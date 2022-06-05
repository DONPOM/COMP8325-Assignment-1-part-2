/**
 * A collection of static-comment-service models
 *
 * <p><b>Require Path:</b> shared/community/collections/static-comment-service</p>
 *
 * @module Shared
 * @submodule Shared.Community
 * @namespace Community.commentCounts
 * @class Collction
 * @constructor
 * @extends BaseCollection
**/
define('shared/community/collections/static-comment-service',[
    'foundation/collections/base-collection',
    'shared/community/models/static-comment-service',
    'foundation/hosts',
    'foundation/helpers/utils'
], function (BaseCollection, staticCommentModel, hosts, utils) {
    'use strict';

    var CommentCounts = BaseCollection.extend({
        url: hosts.s1 + '/ugc/comments/commentData.json',

        model: staticCommentModel,

        parse: function (response) {
            return (response && response.results) ? response.results.assets : [];  // return empty array
        },

        getCommentCountByURL: function (url) {
            return (this.findWhere({assetURL: url})) ? this.findWhere({assetURL: url}).get('commentCount') : -1;
        }

    });

    return CommentCounts;
});

