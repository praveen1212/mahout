package org.apache.mahout.math;

import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.util.Iterator;

public abstract class VectorBinaryAggregate {
  public static final VectorBinaryAggregate[] operations = new VectorBinaryAggregate[] {
      // case 1
      new AggregateNonzerosIterateThisLookupThat(),  // 0

      new AggregateNonzerosIterateThatLookupThis(),  // 1

      // case 2
      new AggregateIterateIntersection(),  // 2

      // case 3
      new AggregateIterateUnionSequential(),  // 3

      new AggregateIterateUnionRandom(),  // 5

      // case 4
      new AggregateAllIterateSequential(),  // 7

      new AggregateAllIterateThisLookupThat(),  // 9
      new AggregateAllIterateThatLookupThis(),  // 11

      new AggregateAllLoop(),  // 14
  };

  public abstract boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc);

  public abstract double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc);

  public abstract double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc);

  public static VectorBinaryAggregate getBestOperation(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
    int bestOperationIndex = -1;
    double bestCost = Double.POSITIVE_INFINITY;
    for (int i = 0; i < operations.length; ++i) {
      if (operations[i].isValid(x, y, fa, fc)) {
        double cost = operations[i].estimateCost(x, y, fa, fc);
        // System.out.printf("%s cost %f\n", operations[i].getClass().toString(), cost);
        if (cost < bestCost) {
          bestCost = cost;
          bestOperationIndex = i;
        }
      }
    }
    // System.out.println();
    return operations[bestOperationIndex];
  }

  public static double aggregateBest(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
    return getBestOperation(x, y, fa, fc).aggregate(x, y, fa, fc);
  }

  public static class AggregateNonzerosIterateThisLookupThat extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isLikeRightPlus() && (fa.isAssociativeAndCommutative() || x.isSequentialAccess())
          && fc.isLikeLeftMult();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.getNumNondefaultElements() * x.getIteratorAdvanceCost() * y.getLookupCost();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Vector.Element xe;
      boolean validResult = false;
      double result = 0;
      double thisResult;
      while (xi.hasNext()) {
        xe = xi.next();
        thisResult = fc.apply(xe.get(), y.getQuick(xe.index()));
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult = true;
        }
      }
      return result;
    }
  }

  public static class AggregateNonzerosIterateThatLookupThis extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isLikeRightPlus() && (fa.isAssociativeAndCommutative() || y.isSequentialAccess())
          && fc.isLikeRightMult();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return y.getNumNondefaultElements() * y.getIteratorAdvanceCost() * x.getLookupCost() * x.getLookupCost();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element ye;
      boolean validResult = false;
      double result = 0;
      double thisResult;
      while (yi.hasNext()) {
        ye = yi.next();
        thisResult = fc.apply(x.getQuick(ye.index()), ye.get());
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult = true;
        }
      }
      return result;
    }
  }

  public static class AggregateIterateIntersection extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isLikeRightPlus() && fc.isLikeMult() && x.isSequentialAccess() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return Math.min(x.getNumNondefaultElements() * x.getIteratorAdvanceCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost());
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element xe = null;
      Vector.Element ye = null;
      boolean advanceThis = true;
      boolean advanceThat = true;
      boolean validResult = false;
      double result = 0;
      double thisResult;
      while (xi.hasNext() && yi.hasNext()) {
        if (advanceThis) {
          xe = xi.next();
        }
        if (advanceThat) {
          ye = yi.next();
        }
        if (xe.index() == ye.index()) {
          thisResult = fc.apply(xe.get(), ye.get());
          if (validResult) {
            result = fa.apply(result, thisResult);
          } else {
            result = thisResult;
            validResult = true;
          }
          advanceThis = true;
          advanceThat = true;
        } else {
          if (xe.index() < ye.index()) { // f(x, 0) = 0
            advanceThis = true;
            advanceThat = false;
          } else { // f(0, y) = 0
            advanceThis = false;
            advanceThat = true;
          }
        }
      }
      return result;
    }
  }

  public static class AggregateIterateUnionSequential extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isLikeRightPlus() && !fc.isDensifying()
          && x.isSequentialAccess() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return Math.max(x.getNumNondefaultElements() * x.getIteratorAdvanceCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost());
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element xe = null;
      Vector.Element ye = null;
      boolean advanceThis = true;
      boolean advanceThat = true;
      boolean validResult = false;
      double result = 0;
      double thisResult;
      while (true) {
        if (advanceThis) {
          if (xi.hasNext()) {
            xe = xi.next();
          } else {
            xe = null;
          }
        }
        if (advanceThat) {
          if (yi.hasNext()) {
            ye = yi.next();
          } else {
            ye = null;
          }
        }
        if (xe != null && ye != null) { // both vectors have nonzero elements
          if (xe.index() == ye.index()) {
            thisResult = fc.apply(xe.get(), ye.get());
            advanceThis = true;
            advanceThat = true;
          } else {
            if (xe.index() < ye.index()) { // f(x, 0)
              thisResult = fc.apply(xe.get(), 0);
              advanceThis = true;
              advanceThat = false;
            } else {
              thisResult = fc.apply(0, ye.get());
              advanceThis = false;
              advanceThat = true;
            }
          }
        } else if (xe != null) { // just the first one still has nonzeros
          thisResult = fc.apply(xe.get(), 0);
          advanceThis = true;
          advanceThat = false;
        } else if (ye != null) { // just the second one has nonzeros
          thisResult = fc.apply(0, ye.get());
          advanceThis = false;
          advanceThat = true;
        } else { // we're done, both are empty
          break;
        }
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult =  true;
        }
      }
      return result;
    }
  }

  public static class AggregateIterateUnionRandom extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isLikeRightPlus() && !fc.isDensifying()
          && (fa.isAssociativeAndCommutative() || (x.isSequentialAccess() && y.isSequentialAccess()));
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return Math.max(x.getNumNondefaultElements() * x.getIteratorAdvanceCost() * y.getLookupCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost() * x.getLookupCost());
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      OpenIntHashSet visited = new OpenIntHashSet();
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Vector.Element xe;
      boolean validResult = false;
      double result = 0;
      double thisResult;
      while (xi.hasNext()) {
        xe = xi.next();
        thisResult = fc.apply(xe.get(), y.getQuick(xe.index()));
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult = true;
        }
        visited.add(xe.index());
      }
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element ye;
      while (yi.hasNext()) {
        ye = yi.next();
        if (!visited.contains(ye.index())) {
          thisResult = fc.apply(x.getQuick(ye.index()), ye.get());
          if (validResult) {
            result = fa.apply(result, thisResult);
          } else {
            result = thisResult;
            validResult = true;
          }
        }
      }
      return result;
    }
  }

  public static class AggregateAllIterateSequential extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.isSequentialAccess() && y.isSequentialAccess() && !x.isDense() && !y.isDense();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return Math.max(x.size() * x.getIteratorAdvanceCost(), y.size() * y.getIteratorAdvanceCost());
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.iterator();
      Iterator<Vector.Element> yi = y.iterator();
      Vector.Element xe;
      boolean validResult = false;
      double result = 0;
      double thisResult;
      while (xi.hasNext() && yi.hasNext()) {
        xe = xi.next();
        thisResult = fc.apply(xe.get(), yi.next().get());
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult = true;
        }
      }
      return result;
    }
  }

  public static class AggregateAllIterateThisLookupThat extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return (fa.isAssociativeAndCommutative() || x.isSequentialAccess())
          && !x.isDense();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.size() * x.getIteratorAdvanceCost() * y.getLookupCost();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.iterator();
      Vector.Element xe;
      boolean validResult = false;
      double result = 0;
      double thisResult;
      while (xi.hasNext()) {
        xe = xi.next();
        thisResult = fc.apply(xe.get(), y.getQuick(xe.index()));
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult = true;
        }
      }
      return result;
    }
  }

  public static class AggregateAllIterateThatLookupThis extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return (fa.isAssociativeAndCommutative() || y.isSequentialAccess())
          && !y.isDense();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return y.size() * y.getIteratorAdvanceCost() * x.getLookupCost();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> yi = y.iterator();
      Vector.Element ye;
      boolean validResult = false;
      double result = 0;
      double thisResult;
      while (yi.hasNext()) {
        ye = yi.next();
        thisResult = fc.apply(x.getQuick(ye.index()), ye.get());
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult = true;
        }
      }
      return result;
    }
  }

  public static class AggregateAllLoop extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return true;
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.size() * x.getLookupCost() * y.getLookupCost();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      double result = fc.apply(x.getQuick(0), y.getQuick(0));
      for (int i = 1; i < x.size(); ++i) {
        result = fa.apply(result, fc.apply(x.getQuick(i), y.getQuick(i)));
      }
      return result;
    }
  }
}
