package edu.umn.cs.recsys.uu;

import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.data.dao.ItemEventDAO;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.event.Rating;
import org.grouplens.lenskit.data.history.History;
import org.grouplens.lenskit.data.history.RatingVectorUserHistorySummarizer;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import org.grouplens.lenskit.vectors.similarity.CosineVectorSimilarity;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.*;


/**
 * User-user item scorer.
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class SimpleUserUserItemScorer extends AbstractItemScorer {
    private final UserEventDAO userDao;
    private final ItemEventDAO itemDao;

    @Inject
    public SimpleUserUserItemScorer(UserEventDAO udao, ItemEventDAO idao) {
        userDao = udao;
        itemDao = idao;
    }

    @Override
    public void score(long user, @Nonnull MutableSparseVector scores) {
        SparseVector userVector = getUserRatingVector(user);

        // TODO Score items for this user using user-user collaborative filtering

        // Calculating mean centered rating vector for the user
        MutableSparseVector tmpusr1 = userVector.mutableCopy();

        for (VectorEntry x1: userVector.fast()){
            tmpusr1.set(x1.getKey(), x1.getValue() - userVector.mean());
        }

        // This is the loop structure to iterate over items to score
        for (VectorEntry e: scores.fast(VectorEntry.State.EITHER)) {
            // Users who have rated that item
            LongSet ls = itemDao.getUsersForItem(e.getKey());
            // Similarities between current user and other users
            MutableSparseVector simi = MutableSparseVector.create(ls);
            // Users who have rated that item
            MutableSparseVector usersrateditem = MutableSparseVector.create(ls);

            Iterator it = ls.iterator();

            // Making a vector out of set values from ls
            while(it.hasNext()){
                long val = (Long)it.next();
                usersrateditem.set(val, getUserRatingVector(val).get(e.getKey()));
            }

            // Map to store mean centered ratings for all users for all movies
            HashMap<Long, MutableSparseVector> allusr = new HashMap<Long, MutableSparseVector>();

            // Subtracting mean rating to get mean centered rating for each user
            for (VectorEntry f: usersrateditem.fast()){
                MutableSparseVector tmpusr = getUserRatingVector(f.getKey()).mutableCopy();
                for (VectorEntry x: getUserRatingVector(f.getKey()).fast()){
                    tmpusr.set(x.getKey(), x.getValue() - getUserRatingVector(f.getKey()).mean());
                }
                allusr.put(f.getKey(), tmpusr);
            }

            // Calculating similarity for each user with the current user
            for (VectorEntry g: usersrateditem.fast()){

                if(g.getKey() == user){
                    continue;
                }

                double sim = new CosineVectorSimilarity().similarity(allusr.get(g.getKey()), tmpusr1);

                simi.set(g.getKey(), sim);
            }

            // Sorting by similarity values
            List<Long> list = simi.keysByValue(true);

            int k = 1;
            double sum1 = 0.0;
            double sum2 = 0.0;

            // Calculating rating for the movie
            for (int i = 0; i < list.size(); i++) {
            // Limiting number of similar users to top 30
                if(k==31){
                    break;
                }

                // Calculating numerator for rating formula
                sum1 = sum1 + simi.get(list.get(i))*allusr.get(list.get(i)).get(e.getKey());
                // Calculating denominator for rating formula
                sum2 = sum2 + Math.abs(simi.get(list.get(i)));

                k++;
            }

            double ratx = getUserRatingVector(user).mean() + sum1/sum2;

            // Populating scores vector with ratings
            scores.set(e.getKey(), ratx);
        }
    }

    /**
     * Get a user's rating vector.
     * @param user The user ID.
     * @return The rating vector.
     */
    private SparseVector getUserRatingVector(long user) {
        UserHistory<Rating> history = userDao.getEventsForUser(user, Rating.class);
        if (history == null) {
            history = History.forUser(user);
        }
        return RatingVectorUserHistorySummarizer.makeRatingVector(history);
    }
}
