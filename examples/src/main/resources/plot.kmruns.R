# Functions for plotting k-means comparison plots.
# The results are a data frame (see kmeans-comparison-nospace.csv) with the
# following columns:
# run,type,time,cluster,distance.mean,distance.sd, (type in {km, bkm, skm,
# bskm, boskm}
# distance.q0,distance.q1,distance.q2,distance.q3,distance.q4,count (quartiles
# and count)

# restrict the data set to the given type and run number
km.restrict <- function(data, type, run) {
    rdata <- data[data$type == type & data$run == run,]
    return(rdata)
}

# plots a boxplot for a given run and type's clustering
km.boxplot <- function(rdata, col='black', add=F, side=4) {
    # build the initial box plot with just the mean
    bp <- boxplot(distance.mean ~ cluster, rdata, plot=F)
    # the number of clusters in the data set
    num.rows <- dim(rdata)[1]
    # populate box plot with the correct margins; the whiskers
    bp$stats <- matrix(c(rdata$distance.q0 + 1, rdata$distance.q1 + 1,
                         rdata$distance.mean + 1, rdata$distance.q3 + 1,
                         rdata$distance.q4 + 1), ncol=num.rows, byrow=T)
    # sets the counts of each type
    bp$n <- rdata$count
    # plots the boxes
    bxp(bp, lwd=1, border=col, add=add, axes=!add)
    # plots the mean and labels the axis
    m <- mean(rdata$distance.mean)
    abline(h=m, col=col)
    axis(side=side, at=m, col=col)
    # sets title and labels if necessary
    if (add == F) {
        title(main=paste('Clusters in run', rdata$run[1], 'of', rdata$type[1]),
              xlab='Cluster index', ylab='Distance from point to centroid')
    }
}

# plots the cluster distances in two kinds of runs (km = kmeans vs. bkm = ball kmeans)
# as points for all the runs of that type, computing the mean cluster distance
# for each type
km.compareplot <- function(data, type1, type2) {
    # restrict the types to the requested ones
    t1data <- data[data$type == type1,]
    t2data <- data[data$type == type2,]
    # get the means of the data sets
    m1 <- mean(t1data$distance.mean)
    m2 <- mean(t2data$distance.mean)
    # start plotting: points, mean lines and axes
    # type 1
    plot(t1data$distance.mean, col=t1data$run+1, pch=19, xlab='', ylab='')
    abline(h=m1)
    axis(side=2, at=m1)
    # type 2
    points(t2data$distance.mean, col=t2data$run+1, xlab='', ylab='')
    abline(h=m2, lty='13')
    axis(side=4, at=m2)
    # legends
    legend('topright', legend=c(paste(type1, 'distances'),
                                paste(type2, 'distances')), pch=c(19, 1))
    legend('topleft', legend=c(paste(type1, 'mean'),
                               paste(type2, 'mean')), lty=c('11', '13'))
    title(main=paste(type1, 'vs', type2), xlab='Cluster index',
          ylab='Average cluster distance')
}

# plots all the different classes of runs as boxplots computing the average in
# each case
km.allplot <- function(akm, cols=c('red', 'pink', 'violet', 'green', 'purple',
                                   'blue', 'light blue'),
                       traintest='all',
                       types=c('bkm', 'boskm', 'bskm', 'km', 'oskm', 'skm',
                               'skm0')) {
    if (traintest == 'train') {
        akm <- akm[akm$is.train=='train',]
    } else if (traintest == 'test') {
        akm <- akm[akm$is.train=='test',]
    }
    # gets the unique types of k-means
    utypes <- sort(unique(as.factor(types)))
    akm <- akm[akm$type %in% utypes, ]
    akm$type <- factor(akm$type)
    # sets the margins to have enough spaces for labels
    par(oma=c(3, 2, 1, 0))
    # plots distances grouped by type on a log-y axis
    boxplot((distance.mean+1) ~ type, akm, log='y', boxfill=cols, lwd=2, pch=19)
    # compute the mean distance for each type of clustering
    i <- 1
    for (type in utypes) {
        axis(side=1, at=i, mean(akm[akm$type==type, 'distance.mean']), outer=T,
             col='white')
        i <- i + 1
    }
    # set the title and labels
    title(main=paste('Clustering techniques compared', traintest),
          xlab='Clustering type / overall mean cluster distance',
          ylab='Mean cluster distance')
}

# reshape the data to plot a scatterplot for train/test average cluster 
# distance mean
km.traintest <- function(data) {
    intermediate <- aggregate(data$distance.mean, by=list(run=data$run,
                                                          is.train=data$is.train,
                                                          type=data$type),
                              mean)
    final <- reshape(intermediate, timevar='is.train', idvar=c('run', 'type'),
                     direction='wide')
    names(final) <- c('run', 'type', 'test', 'train')
    return(final)
}

# plots a scatter plot for test/train average cluster dinstance means across
# the different runs
km.traintestscatter <- function(akm, types=c('bkm', 'boskm', 'bskm', 'km',
                                             'oskm', 'skm', 'skm0')) {
    km <- akm[akm$type %in% types, ]
    km$type <- factor(km$type)
    tt <- km.traintest(km)
    plot(test ~ train, tt, col=tt$type, pch=19)
    legend('topleft', fill=as.numeric(unique(tt$type)),
           legend=unique(tt$type))
}
