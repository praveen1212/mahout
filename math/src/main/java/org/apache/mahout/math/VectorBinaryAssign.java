package org.apache.mahout.math;

import com.google.common.base.Preconditions;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.util.Iterator;

public abstract class VectorBinaryAssign {
  private static final VectorBinaryAssign operations[] = new VectorBinaryAssign[] {
      // case 1
      new AssignNonzerosIterateThisLookupThat(),

      new AssignNonzerosIterateThatLookupThisMergeUpdate(),
      new AssignNonzerosIterateThatLookupThisInplaceUpdate(),

      // case 2
      new AssignIterateIntersection(),

      // case 3
      new AssignIterateUnionSequentialMergeUpdates(),
      new AssignIterateUnionSequentialInplaceUpdates(),

      new AssignIterateUnionRandomMergeUpdates(),
      new AssignIterateUnionRandomInplaceUpdates(),

      // case 4
      new AssignAllIterateSequentialMergeUpdates(),
      new AssignAllIterateSequentialInplaceUpdates(),

      new AssignAllIterateThisLookupThatMergeUpdates(),
      new AssignAllIterateThisLookupThatInplaceUpdates(),
      new AssignAllIterateThatLookupThisMergeUpdates(),
      new AssignAllIterateThatLookupThisInplaceUpdates(),

      new AssignAllRandomMergeUpdates(),
      new AssignAllRandomInplaceUpdates(),
  };

  public abstract boolean isValid(Vector x, Vector y, DoubleDoubleFunction f);

  public abstract double estimateCost(Vector x, Vector y, DoubleDoubleFunction f);

  public abstract Vector assign(Vector x, Vector y, DoubleDoubleFunction f);

  public static Vector assignBest(Vector x, Vector y, DoubleDoubleFunction f) {
    int bestOperationIndex = -1;
    double bestCost = Double.POSITIVE_INFINITY;
    for (int i = 0; i < operations.length; ++i) {
      if (operations[i].isValid(x, y, f)) {
        double cost = operations[i].estimateCost(x, y, f);
        if (cost < bestCost) {
          bestCost = cost;
          bestOperationIndex = i;
        }
      }
    }
    Preconditions.checkArgument(bestOperationIndex >= 0, "No valid operation for vector assign");
    return operations[bestOperationIndex].assign(x, y, f);
  }

  public static class AssignNonzerosIterateThisLookupThat extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return f.isLikeLeftMult();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.getNumNondefaultElements() * x.getIteratorAdvanceCost() * y.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Vector.Element xe;
      Vector.Element ye;
      while (xi.hasNext()) {
        xe = xi.next();
        ye = y.getElement(xe.index());
        xe.set(f.apply(xe.get(), ye.get()));
      }
      return x;
    }
  }

  public static class AssignNonzerosIterateThatLookupThisInplaceUpdate extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return f.isLikeRightPlus();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return y.getNumNondefaultElements() * y.getNumNonZeroElements() * y.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element xe;
      Vector.Element ye;
      while (yi.hasNext()) {
        ye = yi.next();
        xe = x.getElement(ye.index());
        xe.set(f.apply(xe.get(), ye.get()));
      }
      return x;
    }
  }

  public static class AssignNonzerosIterateThatLookupThisMergeUpdate extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return f.isLikeRightPlus() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return y.getNumNondefaultElements() * y.getNumNonZeroElements() * y.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element xe;
      Vector.Element ye;
      double result;
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
      while (yi.hasNext()) {
        ye = yi.next();
        xe = x.getElement(ye.index());
        result =  f.apply(xe.get(), ye.get());
        updates.set(ye.index(), result);
      }
      x.mergeUpdates(updates);
      return x;
    }
  }

  public static class AssignIterateIntersection extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return f.isLikeLeftMult() && f.isLikeRightPlus();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.min(x.getNumNondefaultElements() * x.getIteratorAdvanceCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost());
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element xe = null;
      Vector.Element ye = null;
      boolean advanceThis = true;
      boolean advanceThat = true;
      while (xi.hasNext() && yi.hasNext()) {
        if (advanceThis) {
          xe = xi.next();
        }
        if (advanceThat) {
          ye = yi.next();
        }
        if (xe.index() == ye.index()) {
          xe.set(f.apply(xe.get(), ye.get()));
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
      return x;
    }
  }

  public static class AssignIterateUnionSequentialMergeUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return !f.isDensifying() && x.isSequentialAccess() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.max(x.getNumNondefaultElements() * x.getIteratorAdvanceCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost());
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element xe = null;
      Vector.Element ye = null;
      boolean advanceThis = true;
      boolean advanceThat = true;
      OrderedIntDoubleMapping thisUpdates = new OrderedIntDoubleMapping();
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
            thisUpdates.set(xe.index(), f.apply(xe.get(), ye.get()));
            advanceThis = true;
            advanceThat = true;
          } else {
            if (xe.index() < ye.index()) { // f(x, 0)
              thisUpdates.set(xe.index(), f.apply(xe.get(), 0));
              advanceThis = true;
              advanceThat = false;
            } else {
              thisUpdates.set(ye.index(), f.apply(0, ye.get()));
              advanceThis = false;
              advanceThat = true;
            }
          }
        } else if (xe != null) { // just the first one still has nonzeros
          thisUpdates.set(xe.index(), f.apply(xe.get(), 0));
          advanceThis = true;
          advanceThat = false;
        } else if (ye != null) { // just the second one has nonzeros
          thisUpdates.set(ye.index(), f.apply(0, ye.get()));
          advanceThis = false;
          advanceThat = true;
        } else { // we're done, both are empty
          break;
        }
      }
      x.mergeUpdates(thisUpdates);
      return x;
    }
  }

  public static class AssignIterateUnionSequentialInplaceUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return !f.isDensifying() && x.isSequentialAccess() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.max(x.getNumNondefaultElements() * x.getIteratorAdvanceCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost());
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element xe = null;
      Vector.Element ye = null;
      boolean advanceThis = true;
      boolean advanceThat = true;
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
            xe.set(f.apply(xe.get(), ye.get()));
            advanceThis = true;
            advanceThat = true;
          } else {
            if (xe.index() < ye.index()) { // f(x, 0)
              xe.set(f.apply(xe.get(), 0));
              advanceThis = true;
              advanceThat = false;
            } else {
              x.setQuick(ye.index(), f.apply(0, ye.get()));
              advanceThis = false;
              advanceThat = true;
            }
          }
        } else if (xe != null) { // just the first one still has nonzeros
          xe.set(f.apply(xe.get(), 0));
          advanceThis = true;
          advanceThat = false;
        } else if (ye != null) { // just the second one has nonzeros
          x.setQuick(ye.index(), f.apply(0, ye.get()));
          advanceThis = false;
          advanceThat = true;
        } else { // we're done, both are empty
          break;
        }
      }
      return x;
    }
  }

  public static class AssignIterateUnionRandomMergeUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return !f.isDensifying() && (x.isAddConstantTime() || y.isSequentialAccess());
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.max(x.getNumNondefaultElements() * x.getIteratorAdvanceCost() * y.getLookupCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost() * x.getLookupCost());
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      OpenIntHashSet visited = new OpenIntHashSet();
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Vector.Element xe;
      while (xi.hasNext()) {
        xe = xi.next();
        updates.set(xe.index(), f.apply(xe.get(), y.getQuick(xe.index())));
        visited.add(xe.index());
      }
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element ye;
      while (yi.hasNext()) {
        ye = yi.next();
        if (!visited.contains(ye.index())) {
          updates.set(ye.index(), f.apply(x.getQuick(ye.index()), ye.get()));
        }
      }
      x.mergeUpdates(updates);
      return x;
    }
  }

  public static class AssignIterateUnionRandomInplaceUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return !f.isDensifying() && (x.isAddConstantTime() || y.isSequentialAccess());
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.max(x.getNumNondefaultElements() * x.getIteratorAdvanceCost() * y.getLookupCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost() * x.getLookupCost());
    }
    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      OpenIntHashSet visited = new OpenIntHashSet();
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Vector.Element xe;
      while (xi.hasNext()) {
        xe = xi.next();
        xe.set(f.apply(xe.get(), y.getQuick(xe.index())));
        visited.add(xe.index());
      }
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element ye;
      while (yi.hasNext()) {
        ye = yi.next();
        if (!visited.contains(ye.index())) {
          x.setQuick(ye.index(), f.apply(x.getQuick(ye.index()), ye.get()));
        }
      }
      return x;
    }
  }

  public static class AssignAllIterateSequentialMergeUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.isSequentialAccess() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.size();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.iterator();
      Iterator<Vector.Element> yi = y.iterator();
      Vector.Element xe;
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
      while (xi.hasNext() && yi.hasNext()) {
        xe = xi.next();
        double result = f.apply(xe.get(), yi.next().get());
        updates.set(xe.index(), result);
      }
      x.mergeUpdates(updates);
      return x;
    }
  }

  public static class AssignAllIterateSequentialInplaceUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.isSequentialAccess() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.size();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.iterator();
      Iterator<Vector.Element> yi = y.iterator();
      Vector.Element xe;
      while (xi.hasNext() && yi.hasNext()) {
        xe = xi.next();
        x.setQuick(xe.index(), f.apply(xe.get(), yi.next().get()));
      }
      return x;
    }
  }

  public static class AssignAllIterateThisLookupThatMergeUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return true;
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.size() * y.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.iterator();
      Vector.Element xe;
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
      while (xi.hasNext()) {
        xe = xi.next();
        updates.set(xe.index(), f.apply(xe.get(), y.getQuick(xe.index())));
      }
      x.mergeUpdates(updates);
      return x;
    }
  }

  public static class AssignAllIterateThisLookupThatInplaceUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return true;
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.size() * y.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.iterator();
      Vector.Element xe;
      while (xi.hasNext()) {
        xe = xi.next();
        x.setQuick(xe.index(), f.apply(xe.get(), y.getQuick(xe.index())));
      }
      return x;
    }
  }

  public static class AssignAllIterateThatLookupThisMergeUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return true;
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return y.size() * x.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> yi = y.iterator();
      Vector.Element ye;
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
      while (yi.hasNext()) {
        ye = yi.next();
        updates.set(ye.index(), f.apply(x.getQuick(ye.index()), ye.get()));
      }
      x.mergeUpdates(updates);
      return x;
    }
  }

  public static class AssignAllIterateThatLookupThisInplaceUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return true;
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return y.size() * x.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> yi = y.iterator();
      Vector.Element ye;
      while (yi.hasNext()) {
        ye = yi.next();
        x.setQuick(ye.index(), f.apply(x.getQuick(ye.index()), ye.get()));
      }
      return x;
    }
  }

  public static class AssignAllRandomMergeUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return true;
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.size() * x.getLookupCost() * y.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
      for (int i = 0; i < x.size(); ++i) {
        updates.set(i, f.apply(x.getQuick(i), y.getQuick(i)));
      }
      x.mergeUpdates(updates);
      return x;
    }
  }

  public static class AssignAllRandomInplaceUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return true;
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.size() * x.getLookupCost() * y.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      for (int i = 0; i < x.size(); ++i) {
        x.setQuick(i, f.apply(x.getQuick(i), y.getQuick(i)));
      }
      return x;
    }
  }
}
